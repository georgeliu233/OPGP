import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import numpy as np
import multiprocessing
from multiprocessing import Pool, Process
import argparse
import random

import math
import time
import pandas as pd

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2, scenario_pb2
from google.protobuf import text_format

from shapely.geometry import LineString, Point, Polygon
from shapely.affinity import affine_transform, rotate

import matplotlib.pyplot as plt
import matplotlib as mpl

from functools import partial
from glob import glob
from tqdm import tqdm

from utils.occupancy_grid_utils import create_ground_truth_timestep_grids, \
    create_ground_truth_waypoint_grids, _ego_ground_truth_occupancy
from utils.occupancy_render_utils import pack_trajs, pack_maps, \
    render_roadgraph_tf

from utils.train_utils import *

class Processor:
    def __init__(
            self, 
            height=128,
            width=128,
            pixels_per_meter=1.6,
            hist_len=11,
            future_len=50,
            gap=5,
            num_observed=32,
            num_occluded=6,
            num_map=3,
            map_len=100,
            map_buffer=150,
            ego_map=6,
            ego_map_len=200,
            ego_buffer=300,
            ref_max_len=1000,
            planning_horizon=5,
            dt=0.1,
            data_files='',
            save_dir='',
            cumulative_waypoints='false',
            ol_test=False,
            timestep=199
            ):

            self.height = height
            self.width = width
            self.pixels_per_meter = pixels_per_meter
            self.cumulative_waypoints = cumulative_waypoints

            self.hist_len = hist_len
            self.future_len = future_len
            self.num_observed = num_observed
            self.num_occluded = num_occluded
            self.num_map = num_map
            self.map_len = map_len
            self.map_buffer = map_buffer
            self.ego_map = ego_map
            self.ego_map_len = ego_map_len
            self.ego_buffer = ego_buffer
            self.ref_max_len = ref_max_len

            self.horizon = planning_horizon #s
            self.dt = dt #s

            self.data_files = [data_files]
            self.save_dir = save_dir
            self.gap = gap

            self.ol_test = ol_test
    
            self.timestep = timestep
            print(f'timestep:{self.timestep}')

            self.test_scenario_ids = None

            self.get_config()

    def get_config(self):
        center_x,center_y = int(self.width / 2), int(self.height * 0.75)
        config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        config_text = f"""
            num_past_steps: {self.hist_len - 1}
            num_future_steps: {self.future_len}
            num_waypoints: {self.future_len//5}
            cumulative_waypoints: {self.cumulative_waypoints}
            normalize_sdc_yaw: true
            grid_height_cells: {self.height}
            grid_width_cells: {self.width}
            sdc_y_in_grid: {center_y}
            sdc_x_in_grid: {center_x}
            pixels_per_meter: {self.pixels_per_meter}
            agent_points_per_side_length: 48
            agent_points_per_side_width: 16
            """

        text_format.Parse(config_text, config)
        self.config = config

        input_config = occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig()
        iconfig_text = f"""
            num_past_steps: {self.hist_len - 1}
            num_future_steps: {self.future_len}
            num_waypoints: {self.future_len//5}
            cumulative_waypoints: false
            normalize_sdc_yaw: true
            grid_height_cells: {self.height*2}
            grid_width_cells: {self.width*2}
            sdc_y_in_grid: {center_y + int(self.height*0.5)}
            sdc_x_in_grid: {center_x + int(self.width*0.5)}
            pixels_per_meter: {self.pixels_per_meter}
            agent_points_per_side_length: 48
            agent_points_per_side_width: 16
            """

        text_format.Parse(iconfig_text, input_config)
        self.input_config = input_config
    
    def build_map(self, map_features, dynamic_map_states):
        self.lanes = {}
        self.roads = {}
        self.stop_signs = {}
        self.crosswalks = {}
        self.speed_bumps = {}

        # static map features
        for map in map_features:
            map_type = map.WhichOneof("feature_data")
            map_id = map.id
            map = getattr(map, map_type)

            if map_type == 'lane':
                self.lanes[map_id] = map
            elif map_type == 'road_line' or map_type == 'road_edge':
                self.roads[map_id] = map
            elif map_type == 'stop_sign':
                self.stop_signs[map_id] = map
            elif map_type == 'crosswalk': 
                self.crosswalks[map_id] = map
            elif map_type == 'speed_bump':
                self.speed_bumps[map_id] = map
            else:
                continue

        # dynamic map features
        self.traffic_signals = dynamic_map_states
    
    def map_process(self, traj, num_map, map_len, map_buffer, ind, goal=None):
        '''
        Map point attributes
        self_point (x, y, h), left_boundary_point (x, y, h), right_boundary_pont (x, y, h), speed limit (float),
        self_type (int), left_boundary_type (int), right_boundary_type (int), 
        traffic light (int), stop_point (bool), interpolating (bool), stop_sign (bool)
        '''
        vectorized_map = np.zeros(shape=(num_map, map_len, 17))
        vectorized_crosswalks = np.zeros(shape=(3, 100, 3))
        agent_type = int(traj[-1][-1])

        # get all lane polylines
        lane_polylines = get_polylines(self.lanes)

        # get all road lines and edges polylines
        road_polylines = get_polylines(self.roads)

        # find current lanes for the agent
        ref_lane_ids = find_reference_lanes(agent_type, traj, lane_polylines)

        # find candidate lanes
        ref_lanes = []

        # get current lane's forward lanes
        for curr_lane, start in ref_lane_ids.items():
            candidate = depth_first_search(curr_lane, self.lanes, 
                                           dist=lane_polylines[curr_lane][start:].shape[0], threshold=300)
            ref_lanes.extend(candidate)
        
        if agent_type != 2:
            # find current lanes' left and right lanes
            neighbor_lane_ids = find_neighbor_lanes(ref_lane_ids, traj, self.lanes, lane_polylines)

            # get neighbor lane's forward lanes
            for neighbor_lane, start in neighbor_lane_ids.items():
                candidate = depth_first_search(neighbor_lane, self.lanes, 
                                               dist=lane_polylines[neighbor_lane][start:].shape[0], threshold=300)
                ref_lanes.extend(candidate)
            
            # update reference lane ids
            ref_lane_ids.update(neighbor_lane_ids)

        # remove overlapping lanes
        ref_lanes = remove_overlapping_lane_seq(ref_lanes)
        
        # get traffic light controlled lanes and stop sign controlled lanes
        traffic_light_lanes = {}
        stop_sign_lanes = []

        for signal in self.traffic_signals[ind-1].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self.stop_signs.items():
            stop_sign_lanes.extend(sign.lane)
        
        # add lanes to the array
        added_lanes = 0
        for i, s_lane in enumerate(ref_lanes):
            added_points = 0
            if i > num_map - 1:
                break
            
            # create a data cache
            cache_lane = np.zeros(shape=(map_buffer, 17))

            for lane in s_lane:
                curr_index = ref_lane_ids[lane] if lane in ref_lane_ids else 0
                self_line = lane_polylines[lane][curr_index:]

                if added_points >= map_buffer:
                    break      

                # add info to the array
                for point in self_line:
                    # self_point and type
                    cache_lane[added_points, 0:3] = point
                    cache_lane[added_points, 10] = self.lanes[lane].type

                    # left_boundary_point and type
                    for left_boundary in self.lanes[lane].left_boundaries:
                        left_boundary_id = left_boundary.boundary_feature_id
                        left_start = left_boundary.lane_start_index
                        left_end = left_boundary.lane_end_index
                        left_boundary_type = left_boundary.boundary_type # road line type
                        if left_boundary_type == 0:
                            left_boundary_type = self.roads[left_boundary_id].type + 8 # road edge type
                        
                        if left_start <= curr_index <= left_end:
                            left_boundary_line = road_polylines[left_boundary_id]
                            nearest_point = find_neareast_point(point, left_boundary_line)
                            cache_lane[added_points, 3:6] = nearest_point
                            cache_lane[added_points, 11] = left_boundary_type

                    # right_boundary_point and type
                    for right_boundary in self.lanes[lane].right_boundaries:
                        right_boundary_id = right_boundary.boundary_feature_id
                        right_start = right_boundary.lane_start_index
                        right_end = right_boundary.lane_end_index
                        right_boundary_type = right_boundary.boundary_type # road line type
                        if right_boundary_type == 0:
                            right_boundary_type = self.roads[right_boundary_id].type + 8 # road edge type

                        if right_start <= curr_index <= right_end:
                            right_boundary_line = road_polylines[right_boundary_id]
                            nearest_point = find_neareast_point(point, right_boundary_line)
                            cache_lane[added_points, 6:9] = nearest_point
                            cache_lane[added_points, 12] = right_boundary_type

                    # speed limit
                    cache_lane[added_points, 9] = self.lanes[lane].speed_limit_mph / 2.237

                    # interpolating
                    cache_lane[added_points, 15] = self.lanes[lane].interpolating

                    # traffic_light
                    if lane in traffic_light_lanes.keys():
                        cache_lane[added_points, 13] = traffic_light_lanes[lane][0]
                        if np.linalg.norm(traffic_light_lanes[lane][1:] - point[:2]) < 3:
                            cache_lane[added_points, 14] = True
             
                    # add stop sign
                    if lane in stop_sign_lanes:
                        cache_lane[added_points, 16] = True

                    # count
                    added_points += 1
                    curr_index += 1

                    if added_points >= map_buffer:
                        break             
            
            # scale the lane
            vectorized_map[i] = cache_lane[np.linspace(0, added_points, num=map_len, endpoint=False, dtype=np.int)]
          
            # count
            added_lanes += 1
        
        if goal is not None:
            dist_list = {}
            for i in range(vectorized_map.shape[0]):
                dist_list[i] = np.min(np.linalg.norm(vectorized_map[i ,:, :2] - goal, axis=-1))
            sorted_inds = sorted(dist_list.items(), key=lambda item:item[1])[:self.ego_map]
            sorted_inds = [ind[0] for ind in sorted_inds]
            vectorized_map = vectorized_map[sorted_inds]


        # find surrounding crosswalks and add them to the array
        added_cross_walks = 0
        detection = Polygon([(0, -5), (50, -20), (50, 20), (0, 5)])
        detection = affine_transform(detection, [1, 0, 0, 1, traj[-1][0], traj[-1][1]])
        detection = rotate(detection, traj[-1][2], origin=(traj[-1][0], traj[-1][1]), use_radians=True)

        for _, crosswalk in self.crosswalks.items():
            polygon = Polygon([(point.x, point.y) for point in crosswalk.polygon])
            polyline = polygon_completion(crosswalk.polygon)
            polyline = polyline[np.linspace(0, polyline.shape[0], num=100, endpoint=False, dtype=np.int)]

            if detection.intersects(polygon):
                vectorized_crosswalks[added_cross_walks, :polyline.shape[0]] = polyline
                added_cross_walks += 1
            
            if added_cross_walks >= 3:
                break
        
        #map [3, 100, 17] ; crosswalk [4, 50, 3]
        vectorized_map = vectorized_map[:, 0::2, :]
        vectorized_crosswalks = vectorized_crosswalks[:, 0::2, :]

        return vectorized_map.astype(np.float32), vectorized_crosswalks.astype(np.float32)

    @staticmethod
    def ego_frame_dynamics(v, theta):
        ego_v = v.copy()
        ego_v[0] = v[0] * np.cos(theta) + v[1] * np.sin(theta)
        ego_v[1] = v[1] * np.cos(theta) - v[0] * np.sin(theta)

        return ego_v
    
    def dynamic_state_process(self, ind=11):
        traffic_light_lanes = {}
        stop_sign_lanes = []

        for signal in self.traffic_signals[10].lane_states:
            traffic_light_lanes[signal.lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)
            for lane in self.lanes[signal.lane].entry_lanes:
                traffic_light_lanes[lane] = (signal.state, signal.stop_point.x, signal.stop_point.y)

        for i, sign in self.stop_signs.items():
            stop_sign_lanes.extend(sign.lane)
        
        return traffic_light_lanes, stop_sign_lanes

    def history_ogm_process(self, traj_tensor, valid_tensor, ego_traj, sdc_ids=None):
        timestep_grids, observed_valid, ego_occupancy = create_ground_truth_timestep_grids(traj_tensor, valid_tensor, 
                        ego_traj, self.input_config,flow=True, flow_origin=False,sdc_ids=sdc_ids)

        vehicle_flow = timestep_grids.vehicles.all_flow[:, :, 0, :].numpy().astype(np.int8)
        ped_flow = timestep_grids.pedestrians.all_flow[:, :, 0, :].numpy().astype(np.int8)
        cyc_flow = timestep_grids.cyclists.all_flow[:, :, 0, :].numpy().astype(np.int8)

        hist_flow = np.concatenate([vehicle_flow,ped_flow,cyc_flow],axis=-1)

        hist_vehicles, hist_pedestrians, hist_cyclists = timestep_grids.vehicles, timestep_grids.pedestrians, timestep_grids.cyclists
        hist_v_ogm = tf.concat([hist_vehicles.past_occupancy,hist_vehicles.current_occupancy],axis=-1)
        hist_p_ogm = tf.concat([hist_pedestrians.past_occupancy,hist_pedestrians.current_occupancy],axis=-1)
        hist_c_ogm = tf.concat([hist_cyclists.past_occupancy,hist_cyclists.current_occupancy],axis=-1)

        hist_ogm = tf.stack([hist_v_ogm , hist_p_ogm , hist_c_ogm], axis=-1).numpy().astype(np.bool_)
        ego_mask_hist = ego_occupancy[:, :, :11]
        
        return hist_ogm, hist_flow, ego_mask_hist
    
    def gt_ogm_process(self, traj_tensor, valid_tensor, ego_traj, sdc_ids=None, test=False):
        timestep_grids, observed_valid, ego_occupancy = create_ground_truth_timestep_grids(traj_tensor, valid_tensor, 
                        ego_traj, self.config,flow=True, flow_origin=False, sdc_ids=sdc_ids, test=test)
        if test:
            return None, None, None, observed_valid, None
        true_waypoints = create_ground_truth_waypoint_grids(timestep_grids, self.config, flow=True, flow_origin=False)
        
        gt_obs_v = tf.stack(true_waypoints.vehicles.observed_occupancy,axis=0)
        gt_obs_p = tf.stack(true_waypoints.pedestrians.observed_occupancy,axis=0)
        gt_obs_c = tf.stack(true_waypoints.cyclists.observed_occupancy,axis=0)

        gt_occ_v = tf.stack(true_waypoints.vehicles.occluded_occupancy,axis=0)
        gt_occ_p = tf.stack(true_waypoints.pedestrians.occluded_occupancy,axis=0)
        gt_occ_c = tf.stack(true_waypoints.cyclists.occluded_occupancy,axis=0)

        gt_obs = tf.stack([gt_obs_v, gt_obs_p, gt_obs_c],axis=-1).numpy().astype(np.bool_)
        gt_occ = tf.clip_by_value(gt_occ_v + gt_occ_p + gt_occ_c, 0, 1).numpy().astype(np.bool_)

        gt_flow = tf.stack(true_waypoints.vehicles.flow,axis=0)
        gt_flow_p = tf.stack(true_waypoints.pedestrians.flow,axis=0)
        gt_flow_c = tf.stack(true_waypoints.cyclists.flow,axis=0)

        gt_flow = tf.stack([gt_flow, gt_flow_p, gt_flow_c],axis=-1).numpy().astype(np.int8)
        
        ego_mask = tf.stack([ego_occupancy[:, :, 10 * (i + 2)] for i in range(self.future_len//10)], axis=0).numpy().astype(np.bool_)
   
        return gt_obs, gt_occ, gt_flow, observed_valid, ego_mask
    
    def traj_process(self, traj_tensor, valid_tensor, ego_current, observed_valid, occluded_valid, sdc_ids):
        """
        1. Process trajectories for all agents (pack_trajs)
        2. filter agents in fov (observed(currently present) and occluded)
        3. sort neighbors agents in fov separately (observed, occluded)
        """

        #ego traj
        ego_tensor = traj_tensor[sdc_ids]
        ego_traj, gt_traj = ego_tensor[:self.hist_len], ego_tensor[self.hist_len:, :5] 
        self.current_xyh = [ego_current[0], ego_current[1], ego_current[4]]
        #filtered agents
        observed_ids, occluded_ids = {}, {}
        observed_agents = np.zeros((self.num_observed, self.hist_len, 10))
        if self.ol_test:
            observed_agents = np.zeros((self.num_observed, self.hist_len + self.future_len, 10))
        #x, y coordinates
        observed_valid, occluded_valid = observed_valid, occluded_valid
        for i in range(traj_tensor.shape[0]):
            if i==sdc_ids:
                continue
            if observed_valid[i]==1:
                observed_ids[i] = traj_tensor[i, 10, :2] 
        
        sorted_observed = sorted(observed_ids.items(), 
                                 key=lambda item: np.linalg.norm(item[1] - self.current_xyh[:2]))[:self.num_observed]
        for i, obs in enumerate(sorted_observed):
            if self.ol_test:
                observed_agents[i] = traj_tensor[obs[0], :, :]
            else:
                observed_agents[i] = traj_tensor[obs[0], :self.hist_len, :]
        neighbor_traj = observed_agents
        return ego_traj.astype(np.float32), neighbor_traj.astype(np.float32), gt_traj.astype(np.float32)
    
    def route_process(self, sdc_id, timestep, cur_pos, tracks):
        # find reference paths according to the gt trajectory
        gt_path = tracks[sdc_id].states
        # remove rare cases
        try:
            route = find_route(gt_path, timestep, cur_pos, self.lanes, self.crosswalks, self.traffic_signals)
        except:
            return None

        ref_path = np.array(route, dtype=np.float32)

        if ref_path.shape[0] < 1200:
            repeated_last_point = np.repeat(ref_path[np.newaxis, -1], 1200-ref_path.shape[0], axis=0)
            ref_path = np.append(ref_path, repeated_last_point, axis=0)

        return ref_path


    def occupancy_process(self,traj_tensor, valid_tensor, ego_traj, sdc_ids=None, infer=False):
        """
        process the historical and future occupancy
        """
        timestep_grids, observed_valid, ego_occupancy = create_ground_truth_timestep_grids(traj_tensor, valid_tensor, ego_traj, self.config,flow=True,
        flow_origin=False,sdc_ids=sdc_ids)
        
        hist_vehicles, hist_pedestrians, hist_cyclists = timestep_grids.vehicles, timestep_grids.pedestrians, timestep_grids.cyclists

        hist_v_ogm = tf.concat([hist_vehicles.past_occupancy,hist_vehicles.current_occupancy],axis=-1)
        hist_p_ogm = tf.concat([hist_pedestrians.past_occupancy,hist_pedestrians.current_occupancy],axis=-1)
        hist_c_ogm = tf.concat([hist_cyclists.past_occupancy,hist_cyclists.current_occupancy],axis=-1)
        ego_hist = ego_occupancy[...,:self.hist_len].numpy().astype(np.bool_)

        #[h, w, 3]
        hist_ogm = tf.stack([hist_v_ogm, hist_p_ogm, hist_c_ogm], axis=-1).numpy().astype(np.bool_)

        if infer:
            return hist_ogm, np.zeros((5, 128, 128, 3)), np.zeros((5, 128, 128, 3)), observed_valid.numpy(), None, np.zeros((5, 128, 128, 1)), ego_hist

        true_waypoints = create_ground_truth_waypoint_grids(timestep_grids, self.config,flow=True)
        vehicles, pedestrians, cyclists = true_waypoints.vehicles, true_waypoints.pedestrians, true_waypoints.cyclists

        gt_obs_v = tf.stack(true_waypoints.vehicles.observed_occupancy,axis=0)
        gt_occ_v = tf.stack(true_waypoints.vehicles.occluded_occupancy,axis=0)

        gt_obs_p = tf.stack(true_waypoints.pedestrians.observed_occupancy,axis=0)
        gt_occ_p = tf.stack(true_waypoints.pedestrians.occluded_occupancy,axis=0)

        gt_obs_c = tf.stack(true_waypoints.cyclists.observed_occupancy,axis=0)
        gt_occ_c = tf.stack(true_waypoints.cyclists.occluded_occupancy,axis=0)

        gt_obs = tf.concat([gt_obs_v, gt_obs_p, gt_obs_c] ,axis=-1).numpy().astype(np.bool_)
        gt_occ = tf.concat([gt_occ_v, gt_occ_p, gt_occ_c] ,axis=-1).numpy().astype(np.bool_)

        ego_future = _ego_ground_truth_occupancy(ego_occupancy, self.config)
        ego_future = tf.stack(ego_future, axis=0).numpy().astype(np.bool_)

        return hist_ogm, gt_obs, gt_occ, observed_valid.numpy(), None, ego_future, ego_hist
    
    
    def load_open_loop_files(self):
        path = args.ol_dir
        data = pd.read_csv(path)
        self.test_scenario_ids = set(data['Scenario ID'].to_list())

    def normalize_data(self, ego, neighbors, map_lanes, map_crosswalks, ego_map, ego_crosswalk, 
        ground_truth, goal, ref_line, viz=True,sc_ids='', plan_res=None):
        # get the center and heading (local view)
        center, angle = self.current_xyh[:2], self.current_xyh[2]
        # normalize agent trajectories
        ego[:, :5] = agent_norm(ego, center, angle)
        ground_truth = agent_norm(ground_truth, center, angle)

        for i in range(neighbors.shape[0]):
            if neighbors[i, 10, 0] != 0:
                neighbors[i, :, :5] = agent_norm(neighbors[i, :], center, angle, impute=False)  

        # normalize map points
        for i in range(map_lanes.shape[0]):
            lanes = map_lanes[i]
            crosswalks = map_crosswalks[i]

            for j in range(map_lanes.shape[1]):
                lane = lanes[j]
                if lane[0][0] != 0:
                    lane[:, :9] = map_norm(lane, center, angle)

            for k in range(map_crosswalks.shape[1]):
                crosswalk = crosswalks[k]
                if crosswalk[0][0] != 0:
                    crosswalk[:, :3] = map_norm(crosswalk, center, angle)

        for j in range(ego_map.shape[0]):
            lane = ego_map[j]
            if lane[0][0] != 0:
                lane[:, :9] = map_norm(lane, center, angle)

        for k in range(ego_crosswalk.shape[0]):
            crosswalk = ego_crosswalk[k]
            if crosswalk[0][0] != 0:
                crosswalk[:, :3] = map_norm(crosswalk, center, angle)

        plan_lines = np.zeros_like(ref_line)

        ref_line = ref_line_norm(ref_line, center, angle).astype(np.float32)

        goal = goal_norm(goal, center, angle)

        # visulization
        if viz:
            plt.figure()
            for i in range(map_lanes.shape[0]):
                lanes = map_lanes[i]
                crosswalks = map_crosswalks[i]
                for j in range(map_lanes.shape[1]):
                    lane = lanes[j]
                    if lane[0][0] != 0:
                        centerline = lane[:, 0:2]
                        centerline = centerline[centerline[:, 0] != 0]

                for k in range(map_crosswalks.shape[1]):
                    crosswalk = crosswalks[k]
                    if crosswalk[0][0] != 0:
                        crosswalk = crosswalk[crosswalk[:, 0] != 0]
                        plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=4) # plot crosswalk
            
            for j in range(ego_map.shape[0]):
                lane = ego_map[j]
                if lane[0][0] != 0:
                    centerline = lane[:, 0:2]
                    centerline = centerline[centerline[:, 0] != 0]
                    left = lane[:, 3:5]
                    left = left[left[:, 0] != 0]
                    right = lane[:, 6:8]
                    right = right[right[:, 0] != 0]
                    plt.plot(centerline[:, 0], centerline[:, 1],'k', linewidth=1) # plot centerline

            for k in range(ego_crosswalk.shape[0]):
                crosswalk = ego_crosswalk[k]
                if crosswalk[0][0] != 0:
                    crosswalk = crosswalk[crosswalk[:, 0] != 0]
                    plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=4) # plot crosswalk
                
            
            rect = plt.Rectangle((ego[-1, 0]-ego[-1, 6]/2, ego[-1, 1]-ego[-1, 7]/2), ego[-1, 6], ego[-1, 7], linewidth=2, color='r', alpha=0.6, zorder=3,
                                transform=mpl.transforms.Affine2D().rotate_around(*(ego[-1, 0], ego[-1, 1]), ego[-1, 2]) + plt.gca().transData)
            plt.gca().add_patch(rect)
            plt.plot(ref_line[:, 0][ref_line[:, 0]!=0], ref_line[:, 1][ref_line[:, 0]!=0], 'y', linewidth=2, zorder=4)

            plt.plot(ego[:, 0], ego[:, 1],'royalblue' ,linewidth=3,zorder=3)
            future = ground_truth[ground_truth[:, 0] != 0]
            plt.plot(future[:, 0], future[:, 1], 'r', linewidth=3, zorder=3)
            color_map=['purple','brown','pink','olive','gold','royalblue']

            for i in range(neighbors.shape[0]):
                if neighbors[i, 10, 0] != 0:
                    rect = plt.Rectangle((neighbors[i, 10, 0]-neighbors[i, 10, 6]/2, neighbors[i, 10, 1]-neighbors[i, 10, 7]/2), 
                                          neighbors[i, 10, 6], neighbors[i, 10, 7], linewidth=2, color='m', alpha=0.6, zorder=3,
                                          transform=mpl.transforms.Affine2D().rotate_around(*(neighbors[i, 10, 0], neighbors[i, 10, 1]), neighbors[i, 10, 2]) + plt.gca().transData)
                    plt.gca().add_patch(rect)
                    mask = neighbors[i, :, 0] + neighbors[i, :, 1] != 0 
                    plt.plot(neighbors[i, mask, 0], neighbors[i, mask, 1], 'm', linewidth=1, zorder=3)

            if plan_res is not None:
                plt.plot(plan_res[:, 0], plan_res[:, 1], 'c', linewidth=3,zorder=5)
                plt.scatter(plan_res[9::10, 0], plan_res[9::10, 1], 10, 'c',zorder=5)
            circle = plt.Circle([goal[0], goal[1]],color='r')
            plt.gca().add_patch(circle)
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)
            plt.close()

        return ego, neighbors, map_lanes, map_crosswalks, ego_map, ego_crosswalk ,ref_line, ground_truth, goal, plan_lines

    def data_process(self,vis=False):
        for data_file in self.data_files:
            dataset = tf.data.TFRecordDataset(data_file)
            sample_index = [i for i in range(self.hist_len, self.timestep-self.future_len, self.gap)]
            if len(sample_index) ==0:
                sample_index = [11]
            total_len = len(list(dataset))*len(sample_index)
            print(f"Processing {data_file.split('/')[-1]}", total_len)
            start_time = time.time()
            current = 1
            for data in dataset:
                parsed_data = scenario_pb2.Scenario()
                parsed_data.ParseFromString(data.numpy())
                
                scenario_id = parsed_data.scenario_id
                if self.ol_test: 
                    if scenario_id not in self.test_scenario_ids:
                        continue

                sdc_id = parsed_data.sdc_track_index
                time_len = len(parsed_data.tracks[sdc_id].states)
                self.build_map(parsed_data.map_features, parsed_data.dynamic_map_states)

                traj_tensor, valid_tensor, goal = pack_trajs(parsed_data, self.timestep)
                sdc_id = parsed_data.sdc_track_index
                cnt = 0
                for ind in sample_index:
                    #slice current points
                    traj_window, valid_window = traj_tensor[:, ind-self.hist_len:ind+self.future_len, :], valid_tensor[:, ind-self.hist_len:ind+self.future_len, :]

                    ego_current = traj_tensor[sdc_id, ind]

                    ego_goal_dist = np.linalg.norm(ego_current.numpy()[:2] - np.array(goal))

                    if ego_goal_dist < 3 and cnt < 5:
                        continue
                    
                    traffic_light_lanes, stop_sign_lanes = self.dynamic_state_process()
                    map_dict, org_maps = pack_maps(self.lanes, self.roads, self.crosswalks, traffic_light_lanes, 
                                stop_sign_lanes, self.input_config, [ego_current[0], ego_current[1], ego_current[4]])
                    
                    hist_ogm, hist_flow,ego_ogm = self.history_ogm_process(traj_window, valid_window, ego_current, sdc_id)
                    gt_obs, gt_occ, gt_flow, observed_valid, ego_ogm_gt  = self.gt_ogm_process(traj_window, valid_window, ego_current, sdc_id, False)
                    
                    #vectorized trajs for agents in fovs:
                    rg = render_roadgraph_tf(org_maps).numpy().astype(np.uint8)
            
                    ego_traj, neighbor_traj, gt_traj = self.traj_process(traj_window.numpy(), valid_window.numpy(), ego_current.numpy(), observed_valid, None, sdc_id)
                    ref_line = self.route_process(sdc_id, ind, self.current_xyh, parsed_data.tracks)
                    if ref_line is None:
                        continue
                 
                    #agents map and crosswalks:
                    neighbor_map_lanes = np.zeros((self.num_observed, self.num_map, self.map_len//2, 17))
                    neighbor_map_crosswalks = np.zeros((self.num_observed, 3, 50, 3))
                    ego_map_lane, ego_map_crosswalk = self.map_process(ego_traj, self.ego_map*2, self.ego_map_len, self.ego_buffer, ind, goal)


                    for i in range(neighbor_traj.shape[0]):
                        if neighbor_traj[i, -1, 0] != 0:
                            neighbor_map_lanes[i], neighbor_map_crosswalks[i] = self.map_process(neighbor_traj[i], self.num_map, self.map_len, self.map_buffer, ind)
                    
                    #normalize all
                    self.sc_ids = f'{scenario_id}_{ind}'
                    
                    ego, neighbors, neighbor_map_lanes, neighbor_map_crosswalks, ego_map_lane, ego_map_crosswalk , ref_line, ground_truth, new_goal, plan_lines = self.normalize_data(ego_traj, neighbor_traj, 
                    neighbor_map_lanes, neighbor_map_crosswalks, ego_map_lane,ego_map_crosswalk, gt_traj, goal, ref_line, vis, f'{scenario_id}_{ind}')

                    sys.stdout.write(f"\rProcessing{data_file.split('/')[-1]}|length:{current}/{total_len}|{(time.time()-start_time)/current:>.4f}s/sample")
                    sys.stdout.flush()
                    current += 1
                    cnt += 1

                    filename = self.save_dir + f"{scenario_id}_{ind}.npz"
                    np.savez(filename, ego=ego, neighbors=neighbors, neighbor_map_lanes=neighbor_map_lanes, 
                        ego_map_lane=ego_map_lane,ego_map_crosswalk=ego_map_crosswalk,
                        neighbor_map_crosswalks=neighbor_map_crosswalks, gt_future_states=ground_truth,
                        ref_line=ref_line,goal=new_goal,hist_ogm=hist_ogm, gt_obs=gt_obs, gt_occ=gt_occ,
                        ego_ogm=ego_ogm,ego_ogm_gt=ego_ogm_gt,rg=rg,hist_flow=hist_flow,gt_flow=gt_flow
                    )

def process_ol_test_data(data_files):
    processor = Processor(data_files=data_files,height=128,width=128,gap=10,ref_max_len=1200,ego_map=3,future_len=50,
    save_dir=args.save_dir+'/open_loop_test2/',ol_test=True, timestep=91)
    processor.load_open_loop_files()
    processor.data_process(vis=False) 
    print(f'{data_files}-done!')
    with open(args.save_dir+'ol_log.txt','a') as writer:
        writer.write(data_files+'\n') 

def process_training_data(data_files):
    for data_file in data_files:
        processor = Processor(data_files=data_file,height=128,width=128,gap=10,ref_max_len=1200,ego_map=3,future_len=50,
        save_dir=args.save_dir+'train/', timestep=199)
        processor.data_process(vis=False)
        print(f'{data_file},training_done!')
        with open(args.save_dir+'train_log.txt','a') as writer:
            writer.write(data_file+'\n') 

def process_validation_data(data_files):
    # for data_file in data_files:
    processor = Processor(data_files=data_files,height=128,width=128,gap=10,ref_max_len=1200,ego_map=3,future_len=50,
    save_dir=args.save_dir+'valid/',timestep=91)
    processor.data_process(vis=False) 
    print(f'{data_files},_done!')
    with open(args.save_dir+'val_log.txt','a') as writer:
        writer.write(data_files+'\n') 

def file_slice(files, n):
    return [files[i:i+len(files)//n] for i in range(0, len(files), len(files)//n)]

def process_map(func, files, n):
    sliced_files = file_slice(files, n)

    process_list = []
    for i in range(n):
        p = Process(target=func, args=(sliced_files[i],))
        process_list.append(p)
    for i in range(n):
        process_list[i].start()
    for i in range(n):
        process_list[i].join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int,default=16)
    parser.add_argument("--root_dir", type=str,default='', help='path to load original Waymo Datasets')
    parser.add_argument("--save_dir", type=str,default='', help='path to save processed datasets')
    parser.add_argument("--ol_dir", type=str,default='', help='path for open loop test ids csv (optional)')
    args = parser.parse_args()
    processes = args.processes
    
    #Hint: processing full train set is time consuming, so you may sample a ratio for training
    train_root_dir = args.root_dir + 'training_20s/'
    train_list = glob(train_root_dir+'*')
    process_map(process_training_data, train_list, processes)

    # randomly select half portions of val sets:
    val_root_dir = args.root_dir + 'validation/'
    val_list =[
        val_root_dir + 'validation.tfrecord-' + "%05d" % i + '-of-00150'
        for i in random.sample(range(150), 75)
        ]
    
    with Pool(processes=processes) as p:
        p.map(process_validation_data, val_list)
    
    # process a sample or open-loop test, you may also directly employ the sampled val set for ol-testing
    if args.ol_dir != '':
        full_val_list = glob(val_root_dir + '*')
        with Pool(processes=processes) as p:
            p.map(process_ol_test_data, full_val_list)