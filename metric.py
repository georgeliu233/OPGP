import torch
import torch.nn as nn

from torchmetrics.functional import auroc

import numpy as np
import math

import skimage as ski

def draw_ego_mask(ego, length=5.2860+1, width=2.332+0.5, size=(128, 128), pixels_per_meter=1.6):
    
    b = ego.shape[0]
    dego = ego.detach().cpu().numpy()
    masks = []
    for i in range(b):
        x, y, angle = dego[i, 0],  dego[i, 1], dego[i, 2]
        sin, cos = np.sin(angle), np.cos(angle)
        front_left = [x + length/2*cos - width/2*sin, y + length/2*sin + width/2*cos]
        front_right = [x + length/2*cos + width/2*sin, y + length/2*sin - width/2*cos]
        rear_left = [x - length/2*cos - width/2*sin, y - length/2*sin + width/2*cos]
        rear_right = [x - length/2*cos + width/2*sin, y - length/2*sin - width/2*cos]
        poly_xy = np.array([front_left, front_right, rear_right, rear_left]) #(4, 2)
        # x,y -> h, w
        poly_h =  int(size[0]*0.75) - np.round(poly_xy[:, 0] * pixels_per_meter)
        poly_w =  int(size[1]*0.5) - np.round(poly_xy[:, 1] * pixels_per_meter)
        poly_hw = np.stack([poly_h, poly_w] ,axis=-1)
        mask = ski.draw.polygon2mask(size, poly_hw)
        masks.append(mask)
    masks = np.stack(masks, axis=0)
    masks = torch.tensor(masks).to(ego.device)
    return masks

def plan_metrics(trajectories, ego_future):
    l = ego_future.shape[-2]
    trajectories = trajectories[..., :l, :]
    ego_future_valid = torch.ne(ego_future[..., :2], 0)
    ego_trajectory = trajectories[..., :2] * ego_future_valid[:, None, :, :]
    distance = torch.norm(ego_trajectory[:,:, :, :2] - ego_future[:,None, :, :2], dim=-1)

    ade = distance.mean(-1)
    ade, _  = torch.min(ade,dim=-1)
    egoADE = torch.mean(ade)
    fde = distance[:,:,-1]
    fde, _  = torch.min(fde,dim=-1)
    egoFDE = torch.mean(fde)

    fde3 = distance[:,:,29]
    fde3, _  = torch.min(fde3,dim=-1)
    egoFDE3 = torch.mean(fde3)

    fde1 = distance[:,:,9]
    fde1, _  = torch.min(fde1,dim=-1)
    egoFDE1 = torch.mean(fde1)

    return egoADE.item(), egoFDE.item() ,egoFDE3.item(), egoFDE1.item()

from time import time
def occupancy_metrics(preds, target):
    #preds: [batch, t, h, w] (sigmoid) target[b, t, h, w]
    T = target.shape[1]
    check_time = [1, 3, 5]
    auc_list, iou_list = [], []
    
    for i in range(T):
        auc_list.append(auc_metrics(preds[:, i], target[:, i]))
        iou_list.append(soft_iou(preds[:, i], target[:, i]))
    res_auc, res_iou = [], []
    for t in check_time:
        res_auc.append(torch.mean(torch.stack(auc_list[:t])).item())
        res_iou.append(torch.mean(torch.stack(iou_list[:t])).item())
    return res_auc, res_iou

def all_type_occupancy_metrics(preds, target, n_types=2):
    res_list = []
    for i in range(n_types - 1):
        res_auc, res_iou = occupancy_metrics(preds[:, i], target['gt_obs'][..., 0, i])
        res_list.append([res_auc, res_iou])
    if preds.shape[1]<=3:
        res_auc, res_iou = occupancy_metrics(preds[:, 0], (target['gt_occ'] + target['gt_obs'][..., 0, 0]).clamp(0, 1))
    else:
        res_auc, res_iou = occupancy_metrics(preds[:, 3], target['gt_occ'])
    res_list.append([res_auc, res_iou])
    return res_list

def auc_metrics(inputs, target):
    return auroc(inputs, target.int(), task='binary', thresholds=100)

def soft_iou(inputs, target):
    inputs, target = inputs.reshape(-1), target.reshape(-1)
    intersection = torch.mean(torch.mul(inputs, target))
    T_inputs, T_target = torch.mean(inputs), torch.mean(target)
    soft_iou_score = torch.nan_to_num(intersection / (T_inputs + T_target - intersection) ,0.0)
    return soft_iou_score

def check_dynamics(traj, current_state):
    d_t = 0.1
    diff_xy = torch.diff(traj, dim=1)
    diff_x, diff_y = diff_xy[:,:, 0], diff_xy[:,:, 1]

    v_x, v_y, theta = diff_x / d_t, diff_y/d_t,  np.arctan2(diff_y.cpu().numpy(), diff_x.cpu().numpy() + 1e-6)
    theta = torch.tensor(theta).to(v_x.device)
    lon_speed = v_x * torch.cos(theta) + v_y * torch.sin(theta)
    lat_speed = v_y * torch.cos(theta) - v_x * torch.sin(theta)

    acc = torch.diff(lon_speed,dim=-1) / d_t
    jerk = torch.diff(lon_speed,dim=-1,n=2) / d_t**2
    lat_acc = torch.diff(lat_speed,dim=-1) / d_t

    return torch.mean(torch.abs(acc)).item(), torch.mean(torch.abs(jerk)).item(), torch.mean(torch.abs(lat_acc)).item()

def check_traffic(traj, ref_line, gt_modes):
    b, t, c = ref_line.shape
    red_light = False
    off_route = False

    # project to frenet
    distance_to_ref = torch.cdist(traj[:,:, :2], ref_line[:,:, :2])
    #b, L_ego , s_ref
    s_ego = torch.argmin(distance_to_ref, axis=-1)
    distance_to_route = torch.min(distance_to_ref, axis=-1).values
    off_route = torch.any(distance_to_route > 5, dim=1)

    # get stop point 
    stop_point = torch.argmax(ref_line[:,:,-2].int(),dim=1)
    sig = ref_line[torch.arange(b)[:,None],stop_point.unsqueeze(-1),-1].squeeze(1)
    rl_sig = torch.logical_or(sig==1, torch.logical_or(sig==4, sig==7))
    s_stp = s_ego-stop_point.unsqueeze(-1)
    red_light = torch.logical_and(torch.logical_and(stop_point > 0, torch.any(s_stp > 0,dim=1)), rl_sig)

    return red_light, off_route#.float().mean().item()

def compare_to_gt(ego_metric, gt_metric):
    not_gt_metric = torch.logical_not(gt_metric)
    real_metric = torch.logical_and(ego_metric, not_gt_metric)
    if not_gt_metric.float().sum()==0:
        return not_gt_metric.float().sum().item()
    real_val =  real_metric.float().sum() / not_gt_metric.float().sum()
    return real_val.item()

def flow_epe(outputs, targets):
    if 'gt_flow' not in targets:
        return 0
    target_flow = targets['gt_flow']
    target_flow = target_flow.sum(-1)
    flow_exists = torch.logical_or(torch.ne(target_flow[..., 0], 0), torch.ne(target_flow[..., 1], 0)).float()
    flow_outputs = outputs[:, -2:]
    b, c, t, h, w = flow_outputs.shape
    flow_outputs = flow_outputs.permute(0, 2, 3, 4, 1)
    pred_flow = torch.mul(flow_outputs, flow_exists.unsqueeze(-1))#.permute(0, 2, 3, 4, 1) #[b, t, h, w, 2]
    epe_list = []
    # for i in range(3):
    if torch.sum(flow_exists) > 0:
        flow_epe = torch.sum(torch.norm(pred_flow - target_flow, p=2, dim=-1))/ torch.nan_to_num(torch.sum(flow_exists)/2, 1.0)
        return flow_epe.item()
    else:
        return 0
    # return epe_list

class TrainingMetrics:
    def __init__(self):
        self.ade = []
        self.fde = []
        self.il_loss = []
        self.ogm_loss = []
        self.epe = []
    
    def update(self, traj, score, gt_modes, target, il_loss, ogm_loss, outputs):
        ade, fde, fde3, fde1 = plan_metrics(traj, target['ego_future_states'])
        self.epe.append(flow_epe(outputs, target))
        self.ade.append(ade)
        self.fde.append(fde)

        self.il_loss.append(il_loss.item())
        self.ogm_loss.append(ogm_loss.item())
        return np.mean(self.ade), np.mean(self.fde), np.mean(self.il_loss), np.mean(self.ogm_loss)
    
    def result(self):
        return {
            'T_il_loss':np.mean(self.il_loss),
            'T_ogm_loss':np.mean(self.ogm_loss),
            'T_ade':np.mean(self.ade),
            'T_fde':np.mean(self.fde),
            'epe_v':np.mean(self.epe)
        }
        
class ValidationMetrics:
    def __init__(self):
        self.ade = []
        self.fde = []
        self.fde3 = []
        self.fde1 = []
        self.il_loss = []
        self.ogm_loss = []

        self.ogm_auc = []
        self.ogm_iou = []
        self.epe = []

        self.ogm_auc_p = []
        self.ogm_iou_p = []
        self.epe_p = []

        self.ogm_auc_c = []
        self.ogm_iou_c = []
        self.epe_c = []

        self.occ_auc = []
        self.occ_iou = []
    
    def update(self, traj, score, ogm_pred, gt_modes, target, il_loss, ogm_loss):
        ade, fde, fde3, fde1 = plan_metrics(traj, target['ego_future_states'])
        self.ade.append(ade)
        self.fde.append(fde)
        self.fde3.append(fde3)
        self.fde1.append(fde1)

        epe_list = flow_epe(ogm_pred, target)
        self.epe.append(epe_list)

        self.il_loss.append(il_loss.item())
        self.ogm_loss.append(ogm_loss.item())

        ogm_list = all_type_occupancy_metrics(ogm_pred.sigmoid(), target, 4)
        ogm_auc, ogm_iou, occ_auc, occ_iou = ogm_list[0][0][1], ogm_list[0][1][1], ogm_list[-1][0][1], ogm_list[-1][1][1]
        
        self.ogm_auc.append(ogm_auc)
        self.ogm_iou.append(ogm_iou)
        self.occ_auc.append(occ_auc)
        self.occ_iou.append(occ_iou)

        ogm_auc_p, ogm_iou_p, ogm_auc_c, ogm_iou_c = ogm_list[1][0][1], ogm_list[1][1][1], ogm_list[2][0][1], ogm_list[2][1][1]
        self.ogm_auc_p.append(ogm_auc_p)
        self.ogm_iou_p.append(ogm_iou_p)
        self.ogm_auc_c.append(ogm_auc_c)
        self.ogm_iou_c.append(ogm_iou_c)

        return np.mean(self.ade), np.mean(self.fde), np.mean(self.il_loss), np.mean(self.ogm_loss),\
            np.mean(self.ogm_auc), np.mean(self.ogm_iou), np.mean(self.occ_auc), np.mean(self.occ_iou)
    
    def result(self):
        return {
            'E_il_loss':np.mean(self.il_loss), 
            'E_ogm_loss':np.mean(self.ogm_loss),
            'E_ade':np.mean(self.ade),
            'E_fde_5': np.mean(self.fde),
            'E_fde_3': np.mean(self.fde3),
            'E_fde_1': np.mean(self.fde1),
            'ogm_auc_v':np.mean(self.ogm_auc),
            'ogm_iou_v':np.mean(self.ogm_iou),
            'ogm_auc_p':np.mean(self.ogm_auc_p),
            'ogm_iou_p':np.mean(self.ogm_iou_p),
            'ogm_auc_c':np.mean(self.ogm_auc_c),
            'ogm_iou_c':np.mean(self.ogm_iou_c),
            'occ_auc':np.mean(self.occ_auc),
            'occ_iou':np.mean(self.occ_iou),
            'v_epe':np.mean(self.epe)
        }


class TestingMetrics:
    def __init__(self, config, lite_mode=False):

        self.reset()
        self.config = config
        self.ogm_to_position()
        self.lite_mode = lite_mode
    
    def reset(self):
        self.valid_dict = {
            'fde_1s':[], 'fde_3s':[], 'fde_5s':[], 'ade':[],'collisions_rate':[], 'off_road_rate':[], 'red_light':[],
            'acc':[], 'jerk':[], 'lat_acc':[]
        }
        for m in ['auc','iou']:
            for t in [3 ,5, 8]:
                for ty in ['v','p','c','occ']:
                    self.valid_dict[f'{m}_{t}_{ty}'] = []
    
    def ogm_to_position(self):
        indexes = torch.arange(1, self.config.grid_height_cells + 1)
        widths_indexes = - (indexes - self.config.sdc_y_in_grid - 0.5) / self.config.pixels_per_meter
        heights_indexes =  - (indexes - self.config.sdc_x_in_grid - 0.5) / self.config.pixels_per_meter
        #correspnding (x,y) in dense coordinates (h, w, 2)
        coordinates = torch.stack(torch.meshgrid([widths_indexes, heights_indexes]), dim=-1)#.permute(1, 0, 2)
        self.ogm_coordinates = coordinates
    
    
    def gt_ogm_to_position(self, traj, target, t, current_state, col_thres=3):
        ego_mask = 1- target['gt_ego'][:, t]
        all_occupancies = target['gt_obs'][:, t, :, :, 0, :].sum(-1) + target['gt_occ'][:, t , :, :]
        all_occupancies = all_occupancies * ego_mask
        all_occupancies = all_occupancies.clamp(0, 1).unsqueeze(-1)
        b = all_occupancies.shape[0]
        current_ogm_coordinates = self.ogm_coordinates.unsqueeze(0).expand(b, -1, -1, -1).to(all_occupancies.device)

        b, h, w, c = current_ogm_coordinates.shape
        ego_plan = draw_ego_mask(traj[:, t*10 + 9, :3])

        #filtering the occupied occupancies
        collision_grids = ego_plan *  all_occupancies[..., 0]
        collision_grids = collision_grids.reshape(b, h*w)
        collision_grids = collision_grids.sum(1) > col_thres
        
        return collision_grids.float()

    def collsions_check(self, traj, target, current_state):
        col_list = []
        for t in range(5):
            col_list.append(self.gt_ogm_to_position(traj, target, t, current_state))
        col_list = torch.stack(col_list, dim=1).sum(1) >= 1
        col_rate = col_list
        return col_rate
    
    def update(self, traj, score, ogm_pred, gt_modes, target, current_state):

        ade, fde, fde3, fde1 = plan_metrics(traj, target['ego_future_states'][:, :50, :])

        self.valid_dict['ade'].append(ade)
        self.valid_dict['fde_1s'].append(fde1)
        self.valid_dict['fde_3s'].append(fde3)
        self.valid_dict['fde_5s'].append(fde)

        ogm_list = all_type_occupancy_metrics(ogm_pred, target, 4)

        for i, ty in enumerate(['v','p','c','occ']):
            for j, t in enumerate([3, 5, 8]):
                for k, m in enumerate(['auc', 'iou']):
                    self.valid_dict[f'{m}_{t}_{ty}'].append(ogm_list[i][k][j])
                    # self.valid_dict[f'{m}_{t}_{ty}'].append(0)

        gt_traj = target['ego_future_states'][..., :,:]
        if not self.lite_mode:
            col_rate = self.collsions_check(traj, target, current_state)
            self.valid_dict['collisions_rate'].append(col_rate.float().mean().item())

        acc, jerk, lat_acc = check_dynamics(traj, current_state)

        self.valid_dict['acc'].append(acc)
        self.valid_dict['lat_acc'].append(lat_acc)
        self.valid_dict['jerk'].append(jerk)

        red_light, off_route = check_traffic(traj, target['ref_line'], gt_modes)
        gt_red_light, gt_off_route = check_traffic(gt_traj, target['ref_line'], gt_modes)
        self.valid_dict['red_light'].append(compare_to_gt(red_light, gt_red_light))
        self.valid_dict['off_road_rate'].append(compare_to_gt(off_route, gt_off_route))

    def result(self):
        new_dict = {}
        for k, v in self.valid_dict.items():
            new_dict[k] = np.nanmean(v)
        return new_dict