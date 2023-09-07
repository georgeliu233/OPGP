import tensorflow as tf
import numpy as np
from functools import partial

from .occupancy_render_utils import render_occupancy, render_flow_from_inputs, sample_filter, generate_units, render_ego_occupancy
from waymo_open_dataset.utils.occupancy_flow_grids import TimestepGrids, WaypointGrids,_WaypointGridsOneType

from waymo_open_dataset.protos import occupancy_flow_metrics_pb2
from  waymo_open_dataset.utils import occupancy_flow_data

def create_ground_truth_timestep_grids(
    traj_tensor,
    valid_tensor,
    ego_traj,
    config,
    flow=False,
    flow_origin=False,
    sdc_ids=None,
    test=False
):
    """Renders topdown views of agents over past/current/future time frames.

    Args:
    inputs: Dict of input tensors from the motion dataset.
    config: OccupancyFlowTaskConfig proto message.

    Returns:
    TimestepGrids object holding topdown renders of agents.
    """

    timestep_grids = TimestepGrids()

    unit_x, unit_y = generate_units(config.agent_points_per_side_length, config.agent_points_per_side_width)

    # Occupancy grids.
    sample_func = partial(
        sample_filter,
        traj_tensor=traj_tensor,
        valid_tensor=valid_tensor,
        ego_traj=ego_traj,
        config=config,
        unit_x=unit_x,unit_y=unit_y
                    
    )

    current_sample = sample_func(
        times=['current'],
        include_observed=True,
        include_occluded=True,
    )
    current_occupancy, current_valid = render_occupancy(current_sample, config, sdc_ids=None)
    #[num_agents]
    # print(current_valid.shape)
    current_valid = tf.reduce_max(tf.cast(current_valid,tf.int32),axis=-1)[:,0]

    timestep_grids.vehicles.current_occupancy = current_occupancy.vehicles
    timestep_grids.pedestrians.current_occupancy = current_occupancy.pedestrians
    timestep_grids.cyclists.current_occupancy = current_occupancy.cyclists

    past_sample = sample_func(
        times=['past'],
        include_observed=True,
        include_occluded=True,
    )
    past_occupancy,_ = render_occupancy(past_sample, config, sdc_ids=None)
    timestep_grids.vehicles.past_occupancy = past_occupancy.vehicles
    timestep_grids.pedestrians.past_occupancy = past_occupancy.pedestrians
    timestep_grids.cyclists.past_occupancy = past_occupancy.cyclists

    #[num_agents] presence in fov {AT Present}
    observed_valid = current_valid

    if not test:
      future_sample = sample_func(
          times=['future'],
          include_observed=True,
          include_occluded=False,
      )
      future_obs,_ = render_occupancy(future_sample, config, sdc_ids=None)
      timestep_grids.vehicles.future_observed_occupancy = future_obs.vehicles
      timestep_grids.pedestrians.future_observed_occupancy = future_obs.pedestrians
      timestep_grids.cyclists.future_observed_occupancy = future_obs.cyclists


      future_sample_occ = sample_func(
          times=['future'],
          include_observed=False,
          include_occluded=True,
      )
      future_occ, _ = render_occupancy(future_sample_occ, config, sdc_ids=None)
      # occluded_valid = tf.reduce_max(tf.cast(occ_valid,tf.int32),axis=-1)[:,0]
      timestep_grids.vehicles.future_occluded_occupancy = future_occ.vehicles
      timestep_grids.pedestrians.future_occluded_occupancy = future_occ.pedestrians
      timestep_grids.cyclists.future_occluded_occupancy = future_occ.cyclists

    # All occupancy for flow_origin_occupancy.
    if flow_origin or flow:
      all_sample = sample_func(
          times=['past', 'current', 'future'],
          include_observed=True,
          include_occluded=True,
      )
    if flow_origin:
      all_occupancy,_ = render_occupancy(all_sample, config, sdc_ids=None)
      timestep_grids.vehicles.all_occupancy = all_occupancy.vehicles
      timestep_grids.pedestrians.all_occupancy = all_occupancy.pedestrians
      timestep_grids.cyclists.all_occupancy = all_occupancy.cyclists

    # Flow.
    # NOTE: Since the future flow depends on the current and past timesteps, we
    # need to compute it from [past + current + future] sparse points.
    if flow:
      all_flow = render_flow_from_inputs(all_sample, config, sdc_ids=None)
      timestep_grids.vehicles.all_flow = all_flow.vehicles
      timestep_grids.pedestrians.all_flow = all_flow.pedestrians
      timestep_grids.cyclists.all_flow = all_flow.cyclists
    
    if sdc_ids is not None:
      all_sample = sample_func(
          times=['past', 'current', 'future'],
          include_observed=True,
          include_occluded=False,
      )
      ego_occupancy = render_ego_occupancy(all_sample, sdc_ids, config)
      return timestep_grids, observed_valid, ego_occupancy


    return timestep_grids, observed_valid



def create_ground_truth_waypoint_grids(
    timestep_grids: TimestepGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
    flow_origin: bool=False,
    flow: bool=False, 
) -> WaypointGrids:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    config: OccupancyFlowTaskConfig proto message.

  Returns:
    WaypointGrids object.
  """
  if config.num_future_steps % config.num_waypoints != 0:
    raise ValueError(f'num_future_steps({config.num_future_steps}) must be '
                     f'a multiple of num_waypoints({config.num_waypoints}).')

  true_waypoints = WaypointGrids(
      vehicles=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
      pedestrians=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
      cyclists=_WaypointGridsOneType(
          observed_occupancy=[], occluded_occupancy=[], flow=[]),
  )

  # Observed occupancy.
  _add_ground_truth_observed_occupancy_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)
  # Occluded occupancy.
  _add_ground_truth_occluded_occupancy_to_waypoint_grids(
      timestep_grids=timestep_grids,
      waypoint_grids=true_waypoints,
      config=config)
  # Flow origin occupancy.
  if flow_origin:
    _add_ground_truth_flow_origin_occupancy_to_waypoint_grids(
        timestep_grids=timestep_grids,
        waypoint_grids=true_waypoints,
        config=config)
  # Flow.
  if flow:
    _add_ground_truth_flow_to_waypoint_grids(
        timestep_grids=timestep_grids,
        waypoint_grids=true_waypoints,
        config=config)

  return true_waypoints

def _ego_ground_truth_occupancy(ego_occupancy, config):
  waypoint_size = config.num_future_steps // config.num_waypoints
  future_obs = ego_occupancy[..., config.num_past_steps + 1:]
  gt_ogm = []
  for k in range(config.num_waypoints):
    waypoint_end = (k + 1) * waypoint_size
    if config.cumulative_waypoints:
      waypoint_start = waypoint_end - waypoint_size
      # [batch_size, height, width, waypoint_size]
      segment = future_obs[..., waypoint_start:waypoint_end]
      # [batch_size, height, width, 1]
      waypoint_occupancy = tf.reduce_max(segment, axis=-1, keepdims=True)
    else:
      # [batch_size, height, width, 1]
      waypoint_occupancy = future_obs[..., waypoint_end - 1:waypoint_end]
    gt_ogm.append(waypoint_occupancy)

  return gt_ogm
  

def _add_ground_truth_observed_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # [batch_size, height, width, num_future_steps]
    future_obs = timestep_grids.view(object_type).future_observed_occupancy
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = future_obs[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_occupancy = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_occupancy = future_obs[..., waypoint_end - 1:waypoint_end]
      waypoint_grids.view(object_type).observed_occupancy.append(
          waypoint_occupancy)


def _add_ground_truth_occluded_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future topdowns as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # [batch_size, height, width, num_future_steps]
    future_occ = timestep_grids.view(object_type).future_occluded_occupancy
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = future_occ[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_occupancy = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_occupancy = future_occ[..., waypoint_end - 1:waypoint_end]
      waypoint_grids.view(object_type).occluded_occupancy.append(
          waypoint_occupancy)


def _add_ground_truth_flow_origin_occupancy_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates topdowns as origin occupancies for flow fields.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  waypoint_size = config.num_future_steps // config.num_waypoints
  num_history_steps = config.num_past_steps + 1  # Includes past + current.
  num_future_steps = config.num_future_steps
  if waypoint_size > num_history_steps:
    raise ValueError('If waypoint_size > num_history_steps, we cannot find the '
                     'flow origin occupancy for the first waypoint.')

  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # [batch_size, height, width, num_past_steps + 1 + num_future_steps]
    all_occupancy = timestep_grids.view(object_type).all_occupancy
    # Keep only the section containing flow_origin_occupancy timesteps.
    # First remove `waypoint_size` from the end.  Then keep the tail containing
    # num_future_steps timesteps.
    flow_origin_occupancy = all_occupancy[..., :-waypoint_size]
    # [batch_size, height, width, num_future_steps]
    flow_origin_occupancy = flow_origin_occupancy[..., -num_future_steps:]
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size]
        segment = flow_origin_occupancy[..., waypoint_start:waypoint_end]
        # [batch_size, height, width, 1]
        waypoint_flow_origin = tf.reduce_max(segment, axis=-1, keepdims=True)
      else:
        # [batch_size, height, width, 1]
        waypoint_flow_origin = flow_origin_occupancy[..., waypoint_end -
                                                     1:waypoint_end]
      waypoint_grids.view(object_type).flow_origin_occupancy.append(
          waypoint_flow_origin)


def _add_ground_truth_flow_to_waypoint_grids(
    timestep_grids: TimestepGrids,
    waypoint_grids: WaypointGrids,
    config: occupancy_flow_metrics_pb2.OccupancyFlowTaskConfig,
) -> None:
  """Subsamples or aggregates future flow fields as ground-truth labels.

  Args:
    timestep_grids: Holds topdown renders of agents over time.
    waypoint_grids: Holds topdown waypoints selected as ground-truth labels.
    config: OccupancyFlowTaskConfig proto message.
  """
  num_future_steps = config.num_future_steps
  waypoint_size = config.num_future_steps // config.num_waypoints

  for object_type in occupancy_flow_data.ALL_AGENT_TYPES:
    # num_flow_steps = (num_past_steps + num_futures_steps) - waypoint_size
    # [batch_size, height, width, num_flow_steps, 2]
    flow = timestep_grids.view(object_type).all_flow
    # Keep only the flow tail, containing num_future_steps timesteps.
    # [batch_size, height, width, num_future_steps, 2]
    flow = flow[..., -num_future_steps:, :]
    for k in range(config.num_waypoints):
      waypoint_end = (k + 1) * waypoint_size
      if config.cumulative_waypoints:
        waypoint_start = waypoint_end - waypoint_size
        # [batch_size, height, width, waypoint_size, 2]
        segment = flow[..., waypoint_start:waypoint_end, :]
        # Compute mean flow over the timesteps in this segment by counting
        # the number of pixels with non-zero flow and dividing the flow sum
        # by that number.
        # [batch_size, height, width, waypoint_size, 2]
        occupied_pixels = tf.cast(tf.not_equal(segment, 0.0), tf.float32)
        # [batch_size, height, width, 2]
        num_flow_values = tf.reduce_sum(occupied_pixels, axis=3)
        # [batch_size, height, width, 2]
        segment_sum = tf.reduce_sum(segment, axis=3)
        # [batch_size, height, width, 2]
        mean_flow = tf.math.divide_no_nan(segment_sum, num_flow_values)
        waypoint_flow = mean_flow
      else:
        waypoint_flow = flow[..., waypoint_end - 1, :]
      waypoint_grids.view(object_type).flow.append(waypoint_flow)