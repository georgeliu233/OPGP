import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import logging
import glob

import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
from google.protobuf import text_format

# import torchmetrics
from torchvision.ops.focal_loss import sigmoid_focal_loss

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random


def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())

class DrivingData(Dataset):
    def __init__(self, data_dir, use_flow=False):
        self.data_list = glob.glob(data_dir)
        self.use_flow = use_flow

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx],allow_pickle=True)
        ego = data['ego']
        neighbor = data['neighbors'][:, :11, :]

        neighbor_map_lanes = data['neighbor_map_lanes']
        ego_map_lane = data['ego_map_lane']

        neighbor_crosswalk = data['neighbor_map_crosswalks']
        ego_crosswalk = data['ego_map_crosswalk']
    
        ego_future_states = data['gt_future_states']

        ref_line = data['ref_line']
        goal = data['goal']

        hist_ogm = data['hist_ogm']
        ego_ogm = data['ego_ogm']
        
        gt_obs = data['gt_obs']
        gt_occ = data['gt_occ']
        gt_ego = data['ego_ogm_gt']

        if self.use_flow:
            hist_flow = data['hist_flow']
            gt_flow = data['gt_flow']
            road_graph = data['rg'].astype(np.float32)
            road_graph = np.array(road_graph)

            return ego, neighbor, ego_map_lane, neighbor_map_lanes, ego_crosswalk, neighbor_crosswalk,\
            ego_future_states, ref_line, goal, hist_ogm, ego_ogm, gt_obs, gt_occ, gt_ego, hist_flow, gt_flow, road_graph

        return ego, neighbor, ego_map_lane, neighbor_map_lanes, ego_crosswalk, neighbor_crosswalk,\
            ego_future_states, ref_line, goal, hist_ogm, ego_ogm, gt_obs, gt_occ, gt_ego

def batch_to_dict(batch, local_rank, use_flow=False):
    if use_flow:
        ego, neighbor, ego_map_lane, neighbor_map_lanes, ego_crosswalk, neighbor_crosswalk,\
            ego_future_states, ref_line, goal, hist_ogm, ego_ogm, gt_obs, gt_occ, gt_ego, hist_flow, gt_flow, road_graph = batch
    else:
        ego, neighbor, ego_map_lane, neighbor_map_lanes, ego_crosswalk, neighbor_crosswalk,\
                ego_future_states, ref_line, goal, hist_ogm, ego_ogm, gt_obs, gt_occ, gt_ego = batch
    
    # if not use_flow:
    ego_mask = (1 - ego_ogm.to(local_rank).float().unsqueeze(-1))
    hist_ogm = hist_ogm.to(local_rank).float()*ego_mask
    b, h, w, t, c = hist_ogm.shape
    hist_ogm = hist_ogm.reshape(b, h ,w, t*c)

    input_dict =  {
        'ego_state': ego.to(local_rank).float(),
        'neighbor_state': neighbor.to(local_rank).float(),
        'ego_map_lane': ego_map_lane.to(local_rank).float(),
        'neighbor_map_lanes': neighbor_map_lanes.to(local_rank).float(),
        'ego_map_crosswalk': ego_crosswalk.to(local_rank).float(),
        'neighbor_map_crosswalks': neighbor_crosswalk.to(local_rank).float(),
        'hist_ogm': hist_ogm,
        'ego_ogm': ego_ogm.to(local_rank).float(),
    }
    target_dict = {
        'ref_line': ref_line.to(local_rank).float(),
        'goal': goal.to(local_rank).float(),
        'ego_future_states':ego_future_states[..., [0,1,4]].to(local_rank).float(),
        'gt_obs':gt_obs.to(local_rank).float(),
        'gt_occ':gt_occ.to(local_rank).sum(-1).clamp(0, 1).float(),
        'gt_ego':gt_ego.to(local_rank).float(),
    }
    if not use_flow:
        return input_dict, target_dict 
    else:
        road_graph = road_graph[:, 128:128+256, 128:128+256, :]
        input_dict =  {
            'ego_state': ego.to(local_rank).float(),
            'neighbor_state': neighbor.to(local_rank).float(),
            'ego_map_lane': ego_map_lane.to(local_rank).float(),
            'neighbor_map_lanes': neighbor_map_lanes.to(local_rank).float(),
            'ego_map_crosswalk': ego_crosswalk.to(local_rank).float(),
            'neighbor_map_crosswalks': neighbor_crosswalk.to(local_rank).float(),
            'hist_ogm': hist_ogm,
            'ego_ogm': ego_ogm.to(local_rank).float(),
            'hist_flow': hist_flow.to(local_rank).float(),
            'road_graph': road_graph.to(local_rank).float(),
        }
        target_dict = {
            'ref_line': ref_line.to(local_rank).float(),
            'goal': goal.to(local_rank).float(),
            'ego_future_states':ego_future_states[..., [0,1,4]].to(local_rank).float(),
            'gt_obs':gt_obs.to(local_rank).float(),
            'gt_occ':gt_occ.to(local_rank).sum(-1).clamp(0, 1).float(),
            'gt_ego':gt_ego.to(local_rank).float(),
            'gt_flow':gt_flow.to(local_rank).float()
        }
        return input_dict, target_dict 

def occupancy_loss(outputs, targets, use_flow=False):
    ego_mask = 1-targets['gt_ego']
    gt_obs = targets['gt_obs'][..., 0, :] * ego_mask.unsqueeze(-1)
    target_ogm = gt_obs.permute(0, 4, 1, 2, 3) #[b, c, t, h, w]
    # actors:
    actor_loss: torch.Tensor = 0
    alpha_list = [0.1, 0.01, 0.01]
    for i in range(3):
        ref = outputs[:, i]
        tar = target_ogm[:, i]
        loss = sigmoid_focal_loss(ref, tar, alpha=alpha_list[i], gamma=1, reduction='mean')
        actor_loss += loss
    actor_loss = actor_loss / 3

    #occulsions:
    occ_ref, occ_tar = outputs[:, 3], targets['gt_occ']*ego_mask#[:, 0]
    occ_loss = sigmoid_focal_loss(occ_ref, occ_tar, alpha=0.05, gamma=1, reduction='mean')

    if use_flow:
        flow_loss: torch.Tensor = 0
        target_flow = targets['gt_flow']
        target_flow = target_flow.sum(-1)
        flow_exists = torch.logical_or(torch.ne(target_flow[..., 0], 0), torch.ne(target_flow[..., 1], 0)).float()
        flow_outputs = outputs[:, -2:]
        b, c, t, h, w = flow_outputs.shape
        flow_outputs = flow_outputs.permute(0, 2, 3, 4, 1)     
        pred_flow = torch.mul(flow_outputs, flow_exists.unsqueeze(-1))
        exist_num = torch.nan_to_num(torch.sum(flow_exists)/2, 1.0)
        if exist_num > 0:
            flow_loss += F.smooth_l1_loss(pred_flow, target_flow, reduction='sum') / exist_num
    else:
        flow_loss = None

    return actor_loss, occ_loss, flow_loss

def modal_selections(x, y, mode):
    b, n, t, d = x.shape
    if mode=='fde':
        fde_dist = torch.norm(x[:, :, -1, :2] - y[:, -1, :2].unsqueeze(1).expand(-1, n, -1), dim=-1)
        dist = torch.argmin(fde_dist, dim=-1)
    else:
        # joint ade and fde
        fde_dist = torch.norm(x[:, :, -1, :2] - y[:, -1, :2].unsqueeze(1).expand(-1, n, -1), dim=-1)
        ade_dist = torch.norm(x[:, :, :, :2] - y[:, :, :2].unsqueeze(1).expand(-1, n, -1, -1), dim=-1).mean(-1)
        dist = torch.argmin(0.5*fde_dist + ade_dist, dim=-1)

    return dist


def infer_modal_selection(traj, score, targets, use_planning):
    B = score.shape[0]
    gt_modes = torch.argmax(score, dim=1)
    selected_trajs = traj[torch.arange(B)[:, None], gt_modes.unsqueeze(-1)].squeeze(1)
    return selected_trajs, gt_modes


def imitation_loss(traj, score, targets, use_planning=False):
    gt_future = targets['ego_future_states']
    if isinstance(traj, list):
        il_loss: torch.Tensor = 0
        for tr,sc in zip(traj, score):
            loss, selected_trajs, gt_modes = single_layer_planning_loss(tr, sc, targets)
            il_loss += loss
    else:
        il_loss, selected_trajs, gt_modes = single_layer_planning_loss(traj, score, targets)
    
    return il_loss, selected_trajs, gt_modes
        

def single_layer_planning_loss(traj, score, targets):
    gt_future = targets['ego_future_states']
    p_d = traj.shape[-1]
    gt_future = gt_future[...,:p_d]
    gt_modes = modal_selections(traj, gt_future,mode='joint')
    
    classification_loss = F.cross_entropy(score, gt_modes, label_smoothing=0.2)
    B = traj.shape[0]
    selected_trajs = traj[torch.arange(B)[:, None], gt_modes.unsqueeze(-1)].squeeze(1)
    goal_time = [9, 29, 49]
    
    ade_loss = F.smooth_l1_loss(selected_trajs, gt_future[...,:p_d])
    fde_loss = F.smooth_l1_loss(selected_trajs[...,goal_time,:], gt_future[...,goal_time,:p_d])

    il_loss = ade_loss + 0.5*fde_loss + 2*classification_loss

    return il_loss, selected_trajs, gt_modes


def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

def check_non_contributing_params(model, outputs):
    contrib_params = set()
    all_parameters = set(model.parameters())

    for output in outputs:
        contrib_params.update(get_contributing_params(output))
    print(all_parameters - contrib_params)