import tensorflow as tf

#### Example field definition
# Features of road graph.
roadgraph_features = {
    'roadgraph_samples/dir':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
    'roadgraph_samples/id':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/type':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/valid':
        tf.io.FixedLenFeature([20000, 1], tf.int64, default_value=None),
    'roadgraph_samples/xyz':
        tf.io.FixedLenFeature([20000, 3], tf.float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([128], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([128], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([128, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/speed':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([128, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([128, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([128, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([128, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/speed':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([128, 10], tf.float32, default_value=None),
    'scenario/id':
        tf.io.FixedLenFeature([1], tf.string, default_value=None),
}

# Features of traffic lights.
traffic_light_features = {
    'traffic_light_state/current/state':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/valid':
        tf.io.FixedLenFeature([1, 16], tf.int64, default_value=None),
    'traffic_light_state/current/x':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/y':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/current/z':
        tf.io.FixedLenFeature([1, 16], tf.float32, default_value=None),
    'traffic_light_state/past/state':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/valid':
        tf.io.FixedLenFeature([10, 16], tf.int64, default_value=None),
    'traffic_light_state/past/x':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/y':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
    'traffic_light_state/past/z':
        tf.io.FixedLenFeature([10, 16], tf.float32, default_value=None),
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

# road label
road_label = {1:'LaneCenter-Freeway', 2:'LaneCenter-SurfaceStreet', 3:'LaneCenter-BikeLane', 6:'RoadLine-BrokenSingleWhite',
              7:'RoadLine-SolidSingleWhite', 8:'RoadLine-SolidDoubleWhite', 9:'RoadLine-BrokenSingleYellow', 10:'RoadLine-BrokenDoubleYellow', 
              11:'Roadline-SolidSingleYellow', 12:'Roadline-SolidDoubleYellow', 13:'RoadLine-PassingDoubleYellow', 15:'RoadEdgeBoundary', 
              16:'RoadEdgeMedian', 17:'StopSign', 18:'Crosswalk', 19:'SpeedBump'}

road_line_map = {1:['xkcd:grey', 'solid', 14], 2:['xkcd:grey', 'solid', 14], 3:['xkcd:grey', 'solid', 10], 5:['w', 'solid', 2], 6:['w', 'dashed', 2], 
                 7:['w', 'solid', 2], 8:['w', 'solid', 2], 9:['xkcd:yellow', 'dashed', 4], 10:['xkcd:yellow', 'dashed', 2], 
                 11:['xkcd:yellow', 'solid', 2], 12:['xkcd:yellow', 'solid', 3], 13:['xkcd:yellow', 'dotted', 1.5], 15:['y', 'solid', 4.5], 
                 16:['y', 'solid', 4.5], 17:['r', '.', 40], 18:['b', 'solid', 13], 19:['xkcd:orange', 'solid', 13]}

# traffic light label
light_label = {0:'Unknown', 1:'Arrow_Stop', 2:'Arrow_Caution', 3:'Arrow_Go', 4:'Stop', 5:'Caution', 6:'Go', 7:'Flashing_Stop', 8:'Flashing_Caution'}
light_state_map = {0:'k', 1:'r', 2:'b', 3:'g', 4:'r', 5:'b', 6:'g', 7:'r', 8:'b'}
light_state_map_num = {0:0, 1:1, 2:2, 3:3, 4:1, 5:2, 6:3, 7:1, 8:2}
light_state_rank_map = {0: 1, 1: 4, 2: 3, 3: 2, 4: 4, 5: 3, 6: 2, 7: 4, 8: 3}
light_near_state_map = {0:'black', 1:'darkred', 2:'darkblue', 3:'darkgreen', 4:'darkred', 5:'darkblue', 6:'darkgreen', 7:'darkred', 8:'darkblue'}

def linecolormap(value,m_per_pixel):
    return {'color':value[0], 'linestyle':value[1], 'linewidth': value[2]*m_per_pixel/3}

def light_linecolormap(light_state,value,m_per_pixel):
    return {'color':light_state_map[light_state], 'linestyle':value[1], 'linewidth': value[2]*m_per_pixel/3, 
    'zorder': light_state_rank_map[light_state]}

def lightnear_linecolormap(light_state,value,m_per_pixel):
    return {'color':light_near_state_map[light_state], 'linestyle':value[1], 'linewidth': value[2]*m_per_pixel/3, 
    'zorder': light_state_rank_map[light_state]}

def traffic_light_map(road_type, tl_state, tl_near, m_per_pixel):
    road_val = road_line_map[road_type]
    if tl_near:
        return lightnear_linecolormap(tl_state, road_val, m_per_pixel)
    else:
        return light_linecolormap(tl_state, road_val, m_per_pixel)

from waymo_open_dataset.protos import scenario_pb2

_ObjectType = scenario_pb2.Track.ObjectType
ALL_AGENT_TYPES = [
    _ObjectType.TYPE_VEHICLE,
    _ObjectType.TYPE_PEDESTRIAN,
    _ObjectType.TYPE_CYCLIST,
]

agent_color={
    _ObjectType.TYPE_VEHICLE:'r',
    _ObjectType.TYPE_PEDESTRIAN:'g',
    _ObjectType.TYPE_CYCLIST:'b'
    }
