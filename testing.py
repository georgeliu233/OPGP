
import csv
import argparse
import time
import sys

import torch
from torch import optim
from torch.utils.data import DataLoader

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from google.protobuf import text_format

from model.InteractPlanner import InteractPlanner
from utils.net_utils import *
from metric import TestingMetrics

from planner import Planner


def test_modal_selection(traj, score, targets, level=-1):
    gt_future = targets['ego_future_states']
    if isinstance(traj, list):
        traj, score = traj[level], score[level]
    gt_modes = torch.argmax(score, dim=-1)
    B = traj.shape[0]
    selected_trajs = traj[torch.arange(B)[:, None], gt_modes.unsqueeze(-1)].squeeze(1)
    return selected_trajs, gt_modes
    
def flow_warp(bev_pred, current_ogm, occ=False):
    ogm_pred, pred_flow = bev_pred[:, :4].sigmoid(), bev_pred[:, -2:]
    if not occ:
        ogm_pred = torch.cat([(ogm_pred[:,0] + ogm_pred[:,-1]).clamp(0,1).unsqueeze(1), 
                ogm_pred[:, 1:2], ogm_pred[:, 2:3]],dim=1)

    b, c, t, h, w = pred_flow.shape
    pred_flow = pred_flow.permute(0, 2, 3, 4, 1)
    x = torch.linspace(0, w - 1, w)
    y = torch.linspace(0, h - 1, h)
    grid = torch.stack(torch.meshgrid([x, y])).transpose(1, 2)
    grid = grid.permute(1, 2, 0).unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1, -1).to(local_rank)

    flow_grid = grid + pred_flow + 0.5
    flow_grid =  2 * flow_grid / (h) - 1

    warped_flow = []
    for i in range(flow_grid.shape[1]):
        flow_origin_ogm = current_ogm if i==0 else ogm_pred[:, :, i-1]
        wf = F.grid_sample(flow_origin_ogm, flow_grid[:, i], mode='nearest', align_corners=False)
        warped_flow.append(wf)

    warped_flow = torch.stack(warped_flow, dim=2)
    warped_ogm = ogm_pred * warped_flow
    return warped_ogm

def model_testing(valid_data):

    epoch_metrics = TestingMetrics(config)
    model.eval()
    current = 0
    start_time = time.time()
    size = len(valid_data)

    print(f'Testing....')
    for batch in valid_data:
        # prepare data
        inputs, target = batch_to_dict(batch, local_rank, use_flow)

        # query the model
        with torch.no_grad():
            bev_pred, traj, score = model(inputs)
            selected_trajs, gt_modes = test_modal_selection(traj, score, target, level=0)
            selected_ref = target['ref_line']
            b, h, w, d = inputs['hist_ogm'].shape
            types = inputs['hist_ogm'].reshape(b, h, w, d//3, 3)
            current_ogm = types[:, :, :, -1, :].permute(0, 3, 1, 2)
            type_mask = types[..., -1, :].sum(-2).sum(-2) > 0
            warped_ogm = flow_warp(bev_pred, current_ogm)

            planning_inputs = planner.preprocess(inputs['ego_state'], selected_trajs[:, :50, :2], 
                            selected_ref, warped_ogm[:, :, :5], type_mask, config,left=True)
            xy_plan = planner.plan(planning_inputs, selected_ref, inputs['ego_state'])
        
        epoch_metrics.update(xy_plan, score, warped_ogm, gt_modes, target, inputs['ego_state'][:,-1,:])

        current += args.batch_size
        sys.stdout.write(f'\rVal: [{current:>6d}/{size*args.batch_size:>6d}]|{(time.time()-start_time)/current:>.4f}s/sample')
        sys.stdout.flush()

    print('Calculating Open Loop Planning Results...')  
    print(epoch_metrics.result())
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--batch_size", type=int,default=4)
    parser.add_argument("--dim", type=int,default=256)
    parser.add_argument("--use_flow", type=bool, action='store_true', default=True, 
                    help='whether to use flow warp')
    parser.add_argument("--data_dir", type=str, default='', 
                    help='path to load preprocessed data')
    parser.add_argument("--model_dir", type=str, default='',
                     help='path to load pretrained IL model')

    args = parser.parse_args()
    local_rank = args.local_rank

    config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
    config_text = f"""
        num_past_steps: {10}
        num_future_steps: {50}
        num_waypoints: {5}
        cumulative_waypoints: {'true'}
        normalize_sdc_yaw: true
        grid_height_cells: {128}
        grid_width_cells: {128}
        sdc_y_in_grid: {int(128*0.75)}
        sdc_x_in_grid: {64}
        pixels_per_meter: {1.6}
        agent_points_per_side_length: 48
        agent_points_per_side_width: 16
        """

    text_format.Parse(config_text, config)

    use_flow = args.use_flow

    model = InteractPlanner(config, dim=args.dim, enc_layer=2, heads=8, dropout=0.1,
        timestep=5, decoder_dim=384, fpn_len=2, flow_pred=use_flow)

    local_rank = torch.device('cuda')
    print(local_rank)

    model = model.to(local_rank)

    planner = DiffPlanner(device=local_rank,g_length=1200,g_width=60, horizon=5,test_iters=50)

    assert args.model_dir != '', 'you must load a pretrained weights for OL testing!'
    kw_dict = {}
    for k,v in torch.load(args.model_dir,map_location=torch.device('cpu')).items():
        kw_dict[k[7:]] = v
    model.load_state_dict(kw_dict)
    continue_ep = int(args.model_dir.split('_')[-3]) - 1
    print(f'model loaded!:epoch {continue_ep + 1}')

    test_dataset = DrivingData(args.data_dir + f'*.npz', use_flow=True)

    training_size = len(test_dataset)
    print(f'Length test: {training_size}')

    test_data = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)
    model_testing(test_data)
