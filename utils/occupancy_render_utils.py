import tensorflow as tf
from glob import glob
import dataclasses

import numpy as np
import math

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils.occupancy_flow_renderer import _transform_to_image_coordinates, rotate_points_around_origin
from waymo_open_dataset.utils import occupancy_flow_data
from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from waymo_open_dataset.protos import scenario_pb2

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

from PIL import Image

from .waymo_tf_utils import linecolormap, road_label, road_line_map, traffic_light_map, light_state_map_num

_ObjectType = scenario_pb2.Track.ObjectType

@dataclasses.dataclass
class _SampledPoints:
  """Set of points sampled from agent boxes.

  All fields have shape -
  [batch_size, num_agents, num_steps, num_points] where num_points is
  (points_per_side_length * points_per_side_width).
  """
  # [batch, num_agents, num_steps, points_per_agent].
  x: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  y: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  z: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  agent_type: tf.Tensor
  # [batch, num_agents, num_steps, points_per_agent].
  valid: tf.Tensor

def pack_trajs(parsed_data, time=199):
    #x, y, z, heading, length, width, valid required for occupancy
    tracks = parsed_data.tracks
    traj_tensor = np.zeros((len(tracks), time+1, 10))
    valid_tensor = np.zeros((len(tracks), time+1))
    sdc_id = parsed_data.sdc_track_index
    goal = None

    for i, track in enumerate(tracks):
        object_type = track.object_type
        for j, state in enumerate(track.states):
            if state.valid:
                traj_tensor[i, j] = np.array([state.center_x, state.center_y, state.velocity_x, state.velocity_y,
                                                state.heading, state.center_z, state.length, state.width, state.height, object_type])
                valid_tensor[i, j] = 1
                if i== sdc_id:
                    goal = [state.center_x, state.center_y]

    traj_tensor = tf.convert_to_tensor(traj_tensor, tf.float32)
    valid_tensor = tf.convert_to_tensor(valid_tensor, tf.int32)[...,tf.newaxis]

    return traj_tensor, valid_tensor, goal

def _np_to_img_coordinate(points_x, points_y, config):
    pixels_per_meter = config.pixels_per_meter
    points_x = np.round(points_x * pixels_per_meter) + config.sdc_x_in_grid
    points_y = np.round(-points_y * pixels_per_meter) + config.sdc_y_in_grid

    # Filter out points that are located outside the FOV of topdown map.
    point_is_in_fov = np.logical_and(
        np.logical_and(
            np.greater_equal(points_x, 0), np.greater_equal(points_y, 0)),
        np.logical_and(
            np.less(points_x, config.grid_width_cells),
            np.less(points_y, config.grid_height_cells)))

    return points_x, points_y, point_is_in_fov.astype(np.float32)


def wrap_to_pi(theta):
    return (theta+np.pi) % (2*np.pi) - np.pi


def get_polylines_type(lines, traffic_light_lanes, stop_sign_lanes, config, ego_xyh, polyline_len):
    tl_keys = set(traffic_light_lanes.keys())
    polylines = {}
    org_polylnes = []
    for line in lines.keys():
        types = lines[line].type
        polyline = np.array([(map_point.x, map_point.y) for map_point in lines[line].polyline])
        if polyline.shape[0] <= 2:
            continue
        # rotations:
        x, y = polyline[:, 0] - ego_xyh[0], polyline[:, 1] - ego_xyh[1]
        angle = np.pi/ 2 - ego_xyh[2]
        tx = np.cos(angle) * x - np.sin(angle) * y
        ty = np.sin(angle) * x + np.cos(angle) * y
        new_polyline = np.stack([tx, ty], axis=-1)

        if len(polyline) > 1:
            direction = wrap_to_pi(np.arctan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0]) - angle)
            direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
        else:
            direction = np.array([0])[:, np.newaxis]

        trajs = np.concatenate([new_polyline, direction], axis=-1)

        #(x_img, y_img, in_fov)
        ogm_states = np.zeros((trajs.shape[0], 3))
        points_x, points_y, point_is_in_fov = _np_to_img_coordinate(new_polyline[:,0], new_polyline[:,1], config)
  
        ogm_states = np.stack([points_x, points_y, point_is_in_fov], axis=-1)

        # attrib_states: (type, tl_state, near_tl, stop_sign, sp_limit)
        attrib_states = np.zeros((trajs.shape[0], 5))
        attrib_states[:, 0] = types
        if line in tl_keys:
            attrib_states[:, 1] = light_state_map_num[traffic_light_lanes[line][0]]
            near_tl = np.less_equal(np.linalg.norm(polyline[:, :2] - np.array(traffic_light_lanes[line][1:])[np.newaxis,...], axis=-1), 3).astype(np.float32)
            attrib_states[:, 2] = near_tl
        # add stop sign
        if line in stop_sign_lanes:
            attrib_states[:, 3] = True
        try:
            attrib_states[:, 4] = lines[line].speed_limit_mph / 2.237
        except:
            attrib_states[:, 4] = 0

        ogm_center = np.array([0, (config.sdc_y_in_grid - 0.5*config.grid_height_cells)/config.pixels_per_meter])[np.newaxis,...]
        polyline_traj = np.concatenate((trajs, attrib_states, ogm_states), axis=-1)
        org_polylnes.append(polyline_traj)
        traj_splits = np.array_split(polyline_traj, np.ceil(polyline_traj.shape[0] / polyline_len), axis=0)
        i = 0
        for sub_traj in traj_splits: 

            ade = np.mean(np.linalg.norm(sub_traj[:, :2] - ogm_center, axis=-1))
            polylines[f'{line}_{i}'] = (sub_traj, ade)
 
            i += 1

    return polylines, org_polylnes

def render_roadgraph_tf(rg_tensor):
    if len(rg_tensor)==0:
        print('Warning: RG tensor is 0!')
        return tf.zeros((512, 512, 3))
    rg_tensor = tf.convert_to_tensor(np.concatenate(rg_tensor,axis=0))
    topdown_shape = [512, 512, 1]
    rg_x, rg_y, point_is_in_fov = rg_tensor[:, -3], rg_tensor[:, -2], rg_tensor[:, -1]

    types = rg_tensor[:, 3]
    tl_state = rg_tensor[:, 4]

    should_render_point = tf.cast(point_is_in_fov, tf.bool)
    point_indices = tf.cast(tf.where(should_render_point), tf.int32)
    x_img_coord = tf.gather_nd(rg_x, point_indices)[..., tf.newaxis]
    y_img_coord = tf.gather_nd(rg_y, point_indices)[..., tf.newaxis]

    types = tf.gather_nd(types, point_indices)[..., tf.newaxis]
    tl_state = tf.gather_nd(tl_state, point_indices)[..., tf.newaxis]

    num_points_to_render = point_indices.shape.as_list()[0]

    # [num_points_to_render, 3]
    xy_img_coord = tf.concat(
        [
            # point_indices[:, :1],
            tf.cast(y_img_coord, tf.int32),
            tf.cast(x_img_coord, tf.int32),
        ],
        axis=1,
    )
    gt_values = tf.ones_like(x_img_coord, dtype=tf.float32)

    # [batch_size, grid_height_cells, grid_width_cells, 1]
    rg_viz = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
    # assert_shapes([(rg_viz, topdown_shape)])
    rg_type = tf.math.divide_no_nan(tf.cast(tf.scatter_nd(xy_img_coord, types, topdown_shape),tf.float32) , rg_viz)
    rg_tl_type = tf.math.divide_no_nan(tf.cast(tf.scatter_nd(xy_img_coord, tl_state, topdown_shape),tf.float32) , rg_viz)
    rg_viz = tf.clip_by_value(rg_viz, 0.0, 1.0)
    
    return tf.concat([rg_viz, rg_type, rg_tl_type],axis=-1)


def get_crosswalk_type(lines, traffic_light_lanes, stop_sign_lanes, config, ego_xyh, polyline_len):
    tl_keys = set(traffic_light_lanes.keys())
    polylines = {}
    org_polylnes = []
    # id_list = []
    for line in lines.keys():
        types = 18
        polyline = np.array([(map_point.x, map_point.y) for map_point in lines[line].polygon])
        if polyline.shape[0] <= 2:
            continue
        # rotations:
        x, y = polyline[:, 0] - ego_xyh[0], polyline[:, 1] - ego_xyh[1]
        angle = np.pi/ 2 - ego_xyh[2]
        tx = np.cos(angle) * x - np.sin(angle) * y
        ty = np.sin(angle) * x + np.cos(angle) * y
        new_polyline = np.stack([tx, ty], axis=-1)

        if len(polyline) > 1:
            direction = wrap_to_pi(np.arctan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0]) - angle)
            direction = np.insert(direction, -1, direction[-1])[:, np.newaxis]
        else:
            direction = np.array([0])[:, np.newaxis]

        trajs = np.concatenate([new_polyline, direction], axis=-1)

        #(x_img, y_img, in_fov)
        ogm_states = np.zeros((trajs.shape[0], 3))
        points_x, points_y, point_is_in_fov = _np_to_img_coordinate(new_polyline[:,0], new_polyline[:,1], config)
   
        ogm_states = np.stack([points_x, points_y, point_is_in_fov], axis=-1)

        # attrib_states: (type, tl_state, near_tl, stop_sign, sp_limit)
        attrib_states = np.zeros((trajs.shape[0], 5))
        attrib_states[:, 0] = types
        if line in tl_keys:
            attrib_states[:, 1] = light_state_map_num[traffic_light_lanes[line][0]]
            near_tl = np.less_equal(np.linalg.norm(polyline[:, :2] - np.array(traffic_light_lanes[line][1:])[np.newaxis,...], axis=-1), 3).astype(np.float32)
            attrib_states[:, 2] = near_tl
        # add stop sign
        if line in stop_sign_lanes:
            attrib_states[:, 3] = True
        try:
            attrib_states[:, 4] = lines[line].speed_limit_mph / 2.237
        except:
            attrib_states[:, 4] = 0

        ogm_center = np.array([0, (config.sdc_y_in_grid - 0.5*config.grid_height_cells)/config.pixels_per_meter])[np.newaxis,...]
        polyline_traj = np.concatenate((trajs, attrib_states, ogm_states), axis=-1)
        org_polylnes.append(polyline_traj)
        # org_polylnes[line] = polyline_traj
        traj_splits = np.array_split(polyline_traj, np.ceil(polyline_traj.shape[0] / polyline_len), axis=0)
        i = 0
        
        for sub_traj in traj_splits: 
    
            ade = np.mean(np.linalg.norm(sub_traj[:, :2] - ogm_center, axis=-1))
            polylines[f'{line}_{i}'] = (sub_traj, ade)

            i += 1

    return polylines, org_polylnes


def pack_maps(lanes, roads, crosswalks, traffic_light_lanes, stop_sign_lanes, config, ego_xy, polyline_len=20):
    '''
    inputs: (Dict) lanes, roads, crosswalks
    outputs: dict of polyline-trajs with their ids which are inside fov
    '''
    all_poly_trajs = {}
    org_ploys = []

    lane_polylines, lane_ids  = get_polylines_type(lanes, traffic_light_lanes, stop_sign_lanes, config, ego_xy, polyline_len)
    all_poly_trajs.update(lane_polylines)
    # print([id.shape for id in lane_ids.values()])
    org_ploys.extend(lane_ids)

    roads_polylines, road_ids = get_polylines_type(roads, traffic_light_lanes, stop_sign_lanes, config, ego_xy, polyline_len)
    all_poly_trajs.update(roads_polylines)
    org_ploys.extend(road_ids)

    cw_polylines, cw_ids = get_crosswalk_type(crosswalks, traffic_light_lanes, stop_sign_lanes, config, ego_xy, polyline_len)
    all_poly_trajs.update(cw_polylines)
    org_ploys.extend(cw_ids)

    return all_poly_trajs, org_ploys


def points_sample(
    traj_tensor,
    valid_tensor,
    ego_trajs,
    unit_x,
    unit_y,
    config
    ):

    # traj_tensor, valid_tensor = pack_trajs(parsed_data)

    x, y, z = traj_tensor[...,0:1], traj_tensor[...,1:2], traj_tensor[...,5:6]
    length, width = traj_tensor[...,6:7], traj_tensor[...,7:8]
    bbox_yaw = traj_tensor[...,4:5]

    sdc_x = ego_trajs[0:1][tf.newaxis, tf.newaxis, :]
    sdc_y = ego_trajs[1:2][tf.newaxis, tf.newaxis, :]
    sdc_z = ego_trajs[5:6][tf.newaxis, tf.newaxis, :]

    x = x - sdc_x
    y = y - sdc_y
    z = z - sdc_z

    angle = math.pi / 2 - ego_trajs[4:5][tf.newaxis, tf.newaxis, :]
    x, y = rotate_points_around_origin(x, y, angle)
    bbox_yaw = bbox_yaw + angle

    agent_type = traj_tensor[...,-1][...,tf.newaxis]

    return _sample_points_from_agent_boxes(
        x=x,
        y=y,
        z=z,
        bbox_yaw=bbox_yaw,
        width=width,
        length=length,
        agent_type=agent_type,
        valid=valid_tensor,
        unit_x=unit_x,
        unit_y=unit_y
        # points_per_side_length=config.points_per_side_length,
        # points_per_side_width=config.points_per_side_width,
    )

def sample_filter(
    traj_tensor,
    valid_tensor,
    ego_traj,
    config,
    times,
    unit_x,
    unit_y,
    include_observed=True,
    include_occluded=True
    ):

    b,e = _get_num_steps_from_times(times, config)

    # Sample points from agent boxes over specified time frames.
    # All fields have shape [num_agents, num_steps, points_per_agent].
    sampled_points = points_sample(
        traj_tensor[:,b:e],
        valid_tensor[:,b:e],
        ego_traj,
        unit_x,unit_y,
        config
    )

    agent_valid = tf.cast(sampled_points.valid, tf.bool)
    
    include_all = include_observed and include_occluded
    if not include_all and 'future' in times:
        history_times = ['past', 'current']
        b,e = _get_num_steps_from_times(history_times, config)
        agent_is_observed = valid_tensor[:,b:e]
        # [num_agents, 1, 1]
        agent_is_observed = tf.reduce_max(agent_is_observed, axis=1, keepdims=True)
        agent_is_observed = tf.cast(agent_is_observed, tf.bool)

        if include_observed:
            agent_filter = agent_is_observed
        elif include_occluded:
            agent_filter = tf.logical_not(agent_is_observed)
        else:  # Both observed and occluded are off.
            raise ValueError('Either observed or occluded agents must be requested.')
        agent_valid = tf.logical_and(agent_valid, agent_filter)

    return _SampledPoints(
        x=sampled_points.x,
        y=sampled_points.y,
        z=sampled_points.z,
        agent_type=sampled_points.agent_type,
        valid=agent_valid,
    )

def render_ego_occupancy(
    sampled_points,
    sdc_ids,
    config
    ):
    agent_x = sampled_points.x
    agent_y = sampled_points.y
    agent_type = sampled_points.agent_type
    agent_valid = sampled_points.valid

    # Set up assert_shapes.
    assert_shapes = tf.debugging.assert_shapes
    num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
    topdown_shape = [
        config.grid_height_cells, config.grid_width_cells, num_steps
    ]

    # print(topdown_shape)

    # Transform from world coordinates to topdown image coordinates.
    # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
    agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
        points_x=agent_x,
        points_y=agent_y,
        config=config,
    )
    
    assert_shapes([(point_is_in_fov,
                    [num_agents, num_steps, points_per_agent])])

    # Filter out points from invalid objects.
    agent_valid = tf.cast(agent_valid, tf.bool)
    point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)
    agent_x, agent_y, should_render_point = agent_x[sdc_ids][tf.newaxis,...], agent_y[sdc_ids][tf.newaxis,...], point_is_in_fov_and_valid[sdc_ids][tf.newaxis,...]

    # Collect points for ego vehicle
    assert_shapes([
        (should_render_point,
        [1, num_steps, points_per_agent]),
    ])

    # [num_points_to_render, 4]
    point_indices = tf.cast(tf.where(should_render_point), tf.int32)

    # [num_points_to_render, 1]
    x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
    y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

    num_points_to_render = point_indices.shape.as_list()[0]
    assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                (y_img_coord, [num_points_to_render, 1])])

    # [num_points_to_render, 4]
    xy_img_coord = tf.concat(
        [
            # point_indices[:, :1],
            tf.cast(y_img_coord, tf.int32),
            tf.cast(x_img_coord, tf.int32),
            point_indices[:, 1:2],
        ],
        axis=1,
    )
    # [num_points_to_render]
    gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

    # [batch_size, grid_height_cells, grid_width_cells, num_steps]
    topdown = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)

    
    assert_shapes([(topdown, topdown_shape)])

    # scatter_nd() accumulates values if there are repeated indices.  Since
    # we sample densely, this happens all the time.  Clip the final values.
    topdown = tf.clip_by_value(topdown, 0.0, 1.0)
    return topdown


def render_occupancy(
    sampled_points,
    config,
    sdc_ids=None,
    ):

    agent_x = sampled_points.x
    agent_y = sampled_points.y
    agent_type = sampled_points.agent_type
    agent_valid = sampled_points.valid

    # Set up assert_shapes.
    assert_shapes = tf.debugging.assert_shapes
    num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
    topdown_shape = [
        config.grid_height_cells, config.grid_width_cells, num_steps
    ]

    # print(topdown_shape)

    # Transform from world coordinates to topdown image coordinates.
    # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
    agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
        points_x=agent_x,
        points_y=agent_y,
        config=config,
    )
    assert_shapes([(point_is_in_fov,
                    [num_agents, num_steps, points_per_agent])])

    # Filter out points from invalid objects.
    agent_valid = tf.cast(agent_valid, tf.bool)

    #cases masking the ego car:
    if sdc_ids is not None:
        mask = np.ones((num_agents, 1, 1))
        mask[sdc_ids] = 0
        mask = tf.convert_to_tensor(mask, tf.bool)
        agent_valid = tf.logical_and(agent_valid, mask)

    point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

    occupancies = {}
    for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
        # Collect points for each agent type, i.e., pedestrians and vehicles.
        agent_type_matches = tf.equal(agent_type, object_type)
        should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                            agent_type_matches)

        assert_shapes([
            (should_render_point,
            [num_agents, num_steps, points_per_agent]),
        ])

        # [num_points_to_render, 4]
        point_indices = tf.cast(tf.where(should_render_point), tf.int32)

        # [num_points_to_render, 1]
        x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
        y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

        num_points_to_render = point_indices.shape.as_list()[0]
        assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                    (y_img_coord, [num_points_to_render, 1])])

        # [num_points_to_render, 4]
        xy_img_coord = tf.concat(
            [
                # point_indices[:, :1],
                tf.cast(y_img_coord, tf.int32),
                tf.cast(x_img_coord, tf.int32),
                point_indices[:, 1:2],
            ],
            axis=1,
        )
        # [num_points_to_render]
        gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

        # [batch_size, grid_height_cells, grid_width_cells, num_steps]
        topdown = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)
 
        
        assert_shapes([(topdown, topdown_shape)])

        # scatter_nd() accumulates values if there are repeated indices.  Since
        # we sample densely, this happens all the time.  Clip the final values.
        topdown = tf.clip_by_value(topdown, 0.0, 1.0)
        occupancies[object_type] = topdown

    return occupancy_flow_data.AgentGrids(
        vehicles=occupancies[_ObjectType.TYPE_VEHICLE],
        pedestrians=occupancies[_ObjectType.TYPE_PEDESTRIAN],
        cyclists=occupancies[_ObjectType.TYPE_CYCLIST],
    ), point_is_in_fov_and_valid


def render_flow_from_inputs(
    sampled_points,
    config, 
    sdc_ids=None,
):
  
  agent_x = sampled_points.x
  agent_y = sampled_points.y
  agent_type = sampled_points.agent_type
  agent_valid = sampled_points.valid

  # Set up assert_shapes.
  assert_shapes = tf.debugging.assert_shapes
  num_agents, num_steps, points_per_agent = agent_x.shape.as_list()
  # The timestep distance between flow steps.
  waypoint_size = config.num_future_steps // config.num_waypoints
  num_flow_steps = num_steps - waypoint_size
  topdown_shape = [
      config.grid_height_cells, config.grid_width_cells,num_flow_steps
  ]

  # Transform from world coordinates to topdown image coordinates.
  # All 3 have shape: [batch, num_agents, num_steps, points_per_agent]
  agent_x, agent_y, point_is_in_fov = _transform_to_image_coordinates(
      points_x=agent_x,
      points_y=agent_y,
      config=config,
  )

  # Filter out points from invalid objects.
  agent_valid = tf.cast(agent_valid, tf.bool)
  #cases masking the ego car:
  if sdc_ids is not None:
    mask = np.ones((num_agents, 1, 1))
    mask[sdc_ids] = 0
    mask = tf.convert_to_tensor(mask, tf.bool)
    agent_valid = tf.logical_and(agent_valid, mask)

  # Backward Flow.
  # [num_agents, num_flow_steps, points_per_agent]
  dx = agent_x[:, :-waypoint_size, :] - agent_x[:, waypoint_size:, :]
  dy = agent_y[:, :-waypoint_size, :] - agent_y[:, waypoint_size:, :]

  # Adjust other fields as well to reduce from num_steps to num_flow_steps.
  # agent_x, agent_y: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_x = agent_x[:, waypoint_size:, :]
  agent_y = agent_y[:, waypoint_size:, :]
  # agent_type: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_type = agent_type[:, waypoint_size:, :]
  # point_is_in_fov: Use later timesteps since flow vectors go back in time.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  point_is_in_fov = point_is_in_fov[:, waypoint_size:, :]
  # agent_valid: And the two timesteps.  They both need to be valid.
  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  agent_valid = tf.logical_and(agent_valid[:, waypoint_size:, :],
                               agent_valid[:, :-waypoint_size, :])

  # [batch_size, num_agents, num_flow_steps, points_per_agent]
  point_is_in_fov_and_valid = tf.logical_and(point_is_in_fov, agent_valid)

  flows = {}
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # Collect points for each agent type, i.e., pedestrians and vehicles.
    agent_type_matches = tf.equal(agent_type, object_type)
    should_render_point = tf.logical_and(point_is_in_fov_and_valid,
                                         agent_type_matches)

    # [batch_size, height, width, num_flow_steps, 2]
    flow = _render_flow_points_for_one_agent_type(
        agent_x=agent_x,
        agent_y=agent_y,
        dx=dx,
        dy=dy,
        should_render_point=should_render_point,
        topdown_shape=topdown_shape,
    )
    flows[object_type] = flow

  return occupancy_flow_data.AgentGrids(
      vehicles=flows[_ObjectType.TYPE_VEHICLE],
      pedestrians=flows[_ObjectType.TYPE_PEDESTRIAN],
      cyclists=flows[_ObjectType.TYPE_CYCLIST],
  )


def _render_flow_points_for_one_agent_type(
    agent_x,
    agent_y,
    dx,
    dy,
    should_render_point,
    topdown_shape,
):
  assert_shapes = tf.debugging.assert_shapes

  # Scatter points across topdown maps for each timestep.  The tensor
  # `point_indices` holds the indices where `should_render_point` is True.
  # It is a 2-D tensor with shape [n, 3], where n is the number of valid
  # agent points inside FOV.  Each row in this tensor contains indices over
  # the following 3 dimensions: (agent, timestep, point).

  # [num_points_to_render, 3]
  point_indices = tf.cast(tf.where(should_render_point), tf.int32)
  # [num_points_to_render, 1]
  x_img_coord = tf.gather_nd(agent_x, point_indices)[..., tf.newaxis]
  y_img_coord = tf.gather_nd(agent_y, point_indices)[..., tf.newaxis]

  num_points_to_render = point_indices.shape.as_list()[0]
  assert_shapes([(x_img_coord, [num_points_to_render, 1]),
                 (y_img_coord, [num_points_to_render, 1])])

  # [num_points_to_render, 4]
  xy_img_coord = tf.concat(
      [
          tf.cast(y_img_coord, tf.int32),
          tf.cast(x_img_coord, tf.int32),
          point_indices[:, 1:2],
      ],
      axis=1,
  )
  # [num_points_to_render]
  gt_values_dx = tf.gather_nd(dx, point_indices)
  gt_values_dy = tf.gather_nd(dy, point_indices)

  gt_values = tf.squeeze(tf.ones_like(x_img_coord, dtype=tf.float32), axis=-1)

  # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps]
  flow_x = tf.scatter_nd(xy_img_coord, gt_values_dx, topdown_shape)
  flow_y = tf.scatter_nd(xy_img_coord, gt_values_dy, topdown_shape)
  num_values_per_pixel = tf.scatter_nd(xy_img_coord, gt_values, topdown_shape)

  # Undo the accumulation effect of tf.scatter_nd() for repeated indices.
  flow_x = tf.math.divide_no_nan(flow_x, num_values_per_pixel)
  flow_y = tf.math.divide_no_nan(flow_y, num_values_per_pixel)

  # [batch_size, grid_height_cells, grid_width_cells, num_flow_steps, 2]
  flow = tf.stack([flow_x, flow_y], axis=-1)
  return flow


def generate_units(points_per_side_length, points_per_side_width):

    if points_per_side_length < 1:
        raise ValueError('points_per_side_length must be >= 1')
    if points_per_side_width < 1:
        raise ValueError('points_per_side_width must be >= 1')

    # Create sample points on a unit square or boundary depending on flag.
    if points_per_side_length == 1:
        step_x = 0.0
    else:
        step_x = 1.0 / (points_per_side_length - 1)
    if points_per_side_width == 1:
        step_y = 0.0
    else:
        step_y = 1.0 / (points_per_side_width - 1)
    unit_x = []
    unit_y = []
    for xi in range(points_per_side_length):
        for yi in range(points_per_side_width):
            unit_x.append(xi * step_x - 0.5)
            unit_y.append(yi * step_y - 0.5)

    # Center unit_x and unit_y if there was only 1 point on those dimensions.
    if points_per_side_length == 1:
        unit_x = np.array(unit_x) + 0.5
    if points_per_side_width == 1:
        unit_y = np.array(unit_y) + 0.5

    unit_x = tf.convert_to_tensor(unit_x, tf.float32)
    unit_y = tf.convert_to_tensor(unit_y, tf.float32)

    return unit_x, unit_y


def _sample_ego_from_boxes(x, y, bbox_yaw, width, length, unit_x, unit_y):
    sin_yaw = tf.sin(bbox_yaw)
    cos_yaw = tf.cos(bbox_yaw)

    tx = cos_yaw * length * unit_x - sin_yaw * width * unit_y + x
    ty = sin_yaw * length * unit_x + cos_yaw * width * unit_y + y
    return tx, ty


def _sample_points_from_agent_boxes(
    x, y, z, bbox_yaw, width, length, agent_type, valid, unit_x, unit_y
    ):
    assert_shapes = tf.debugging.assert_shapes
    assert_shapes([(x, [..., 1])])
    x_shape = x.get_shape().as_list()

    # Transform the unit square points to agent dimensions and coordinate frames.
    sin_yaw = tf.sin(bbox_yaw)
    cos_yaw = tf.cos(bbox_yaw)

    # [..., num_points]
    tx = cos_yaw * length * unit_x - sin_yaw * width * unit_y + x
    ty = sin_yaw * length * unit_x + cos_yaw * width * unit_y + y
    tz = tf.broadcast_to(z, tx.shape)

    # points_shape = x_shape[:-1] + [num_points]
    agent_type = tf.broadcast_to(agent_type, tx.shape)
    valid = tf.broadcast_to(valid, tx.shape)

    return _SampledPoints(x=tx, y=ty, z=tz, agent_type=agent_type, valid=valid)


def _get_num_steps_from_times(
    times,
    config):
  """Returns number of timesteps that exist in requested times."""
  p, c, f = config.num_past_steps, 1, config.num_future_steps
  dict_1 = {'past':(0, p), 'current':(p, p+c), 'future':(p+c, p+c+f)}
  if len(times)==0:
    raise NotImplementedError()
  elif len(times)==1:
    return dict_1[times[0]]
  elif len(times)==2:
      assert times[0]=='past'
      return (0, p + c)
  else:
      return (0, p + c + f)

