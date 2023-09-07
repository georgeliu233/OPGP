import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1, 0, 2)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class AgentEncoder(nn.Module):
    def __init__(self):
        super(AgentEncoder, self).__init__()
        self.motion = nn.LSTM(9, 256, 2, batch_first=True)
        self.type_embed = nn.Embedding(3, 256, padding_idx=0)

    def forward(self, inputs):
        traj, _ = self.motion(inputs[...,:-1])
        types = inputs[...,0,-1].int().clamp(0, 2)
        types = self.type_embed(types)
        output = traj[:, -1] + types
        return output

class LaneEncoder(nn.Module):
    def __init__(self):
        super(LaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.left_line = nn.Linear(3, 128)
        self.right_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.self_type = nn.Embedding(4, 64, padding_idx=0)
        self.left_type = nn.Embedding(11, 64, padding_idx=0)
        self.right_type = nn.Embedding(11, 64, padding_idx=0)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)
        self.interpolating = nn.Embedding(2, 64)
        self.stop_sign = nn.Embedding(2, 64)
        self.stop_point = nn.Embedding(2, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(512, 384), nn.ReLU(), nn.Linear(384, 256))
        self.position_encode = PositionalEncoding(max_len=100)

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        left_line = self.left_line(inputs[..., 3:6])
        right_line = self.right_line(inputs[...,  6:9])
        speed_limit = self.speed_limit(inputs[..., 9].unsqueeze(-1))
        self_type = self.self_type(inputs[..., 10].int().clamp(0, 3))
        left_type = self.left_type(inputs[..., 11].int().clamp(0, 10))
        right_type = self.right_type(inputs[..., 12].int().clamp(0, 10)) 
        traffic_light = self.traffic_light_type(inputs[..., 13].int().clamp(0, 8))
        stop_point = self.stop_point(inputs[..., 14].int().clamp(0, 1))
        interpolating = self.interpolating(inputs[..., 15].int().clamp(0, 1)) 
        stop_sign = self.stop_sign(inputs[..., 16].int().clamp(0, 1))

        lane_attr = self_type + left_type + right_type + traffic_light + stop_point + interpolating + stop_sign
        lane_embedding = torch.cat([self_line, left_line, right_line, speed_limit, lane_attr], dim=-1)
    
        # process
        output = self.pointnet(lane_embedding)
        output = self.position_encode(output)

        return output

class NeighborLaneEncoder(nn.Module):
    def __init__(self):
        super(NeighborLaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.speed_limit = nn.Linear(1, 64)
        self.traffic_light_type = nn.Embedding(9, 64, padding_idx=0)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256))
        self.position_encode = PositionalEncoding(max_len=50)

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])
        speed_limit = self.speed_limit(inputs[..., 3].unsqueeze(-1))
        traffic_light = self.traffic_light_type(inputs[..., 4].int().clamp(0, 8))
        lane_embedding = torch.cat([self_line, speed_limit, traffic_light], dim=-1)
    
        # process
        output = self.pointnet(lane_embedding)
        output = self.position_encode(output)

        return output

class CrosswalkEncoder(nn.Module):
    def __init__(self):
        super(CrosswalkEncoder, self).__init__()
        self.pointnet = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256)
        )
        self.position_encode = PositionalEncoding(max_len=50)
    
    def forward(self, inputs):
        output = self.pointnet(inputs)
        output = self.position_encode(output)
        return output

class VectorEncoder(nn.Module):
    def __init__(self, dim=256, layers=4, heads=8, dropout=0.1):
        super(VectorEncoder, self).__init__()

        self.ego_encoder = AgentEncoder()
        self.neighbor_encoder = AgentEncoder()

        # self.map_encoder = NeighborLaneEncoder()
        self.ego_map_encoder = LaneEncoder()
        self.crosswalk_encoder = CrosswalkEncoder()

        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                         activation='gelu', dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers)
    
    def segment_map(self, map, map_encoding):
        B, N_r, N_p, D = map_encoding.shape 
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)
        
        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_r, N_p//10, -1)
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)
        map_mask[:, 0] = False # prevent nan

        return map_encoding, map_mask
    
    def forward(self, inputs):

        ego = inputs['ego_state']
        neighbor = inputs['neighbor_state']

        actors = torch.cat([ego.unsqueeze(1), neighbor], dim=1)
        actors_mask = torch.eq(actors, 0)[:, :, -1, 0]
        actors_mask[:, 0] = False

        ego = self.ego_encoder(ego)
        B, N, T, D = neighbor.shape
        neighbor = self.neighbor_encoder(neighbor.reshape(B*N, T, D))
        neighbor = neighbor.reshape(B, N, -1)

        encode_actors = torch.cat([ego.unsqueeze(1), neighbor], dim=1)
        B,N,C = encode_actors.shape

        ego_maps = inputs['ego_map_lane']
        ego_encode_maps = self.ego_map_encoder(ego_maps)
        B, M, L, D = ego_maps.shape
        ego_encode_maps, ego_map_mask = self.segment_map(ego_maps, ego_encode_maps) #(B*N,N_map,D)
        encode_maps = ego_encode_maps
        map_mask = ego_map_mask

        crosswalks = inputs['ego_map_crosswalk']
        encode_cws = self.crosswalk_encoder(crosswalks)
        B, M, L, D = crosswalks.shape
        encode_cws, cw_mask = self.segment_map(crosswalks, encode_cws)

        encode_maps = torch.cat([encode_maps, encode_cws], dim=1)
        map_mask = torch.cat([map_mask, cw_mask], dim=1)

        encode_inputs = torch.cat([encode_actors, encode_maps], dim=1) #(B*N,N + N_map + N_cw, D)
        encode_masks = torch.cat([actors_mask, map_mask], dim=1) #(B*N,N + N_map + N_cw)
        encode_masks[:,0] = False

        encodings = self.fusion_encoder(encode_inputs ,src_key_padding_mask=encode_masks)

        _, L, D = encodings.shape
        N = 1
        encodings = encodings.reshape(B, N, L, D)
        encode_masks = encode_masks.reshape(B, N, L)

        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': encode_masks
        }
        
        encoder_outputs.update(inputs)
        return encoder_outputs


class PredLaneEncoder(nn.Module):
    def __init__(self):
        super(PredLaneEncoder, self).__init__()
        # encdoer layer
        self.self_line = nn.Linear(3, 128)
        self.map_flow = nn.Linear(3, 128)

        self.speed_limit = nn.Linear(1, 64)
        self.traffic_light_type = nn.Embedding(4, 64, padding_idx=0)
        self.self_type = nn.Embedding(20, 64, padding_idx=0)
        self.stop_sign = nn.Linear(1, 64)

        # hidden layers
        self.pointnet = nn.Sequential(nn.Linear(384, 256), nn.ReLU(), nn.Linear(256, 256))
        self.position_encode = PositionalEncoding(max_len=20)

    def forward(self, inputs):
        # embedding
        self_line = self.self_line(inputs[..., :3])

        road_type = self.self_type(inputs[..., 3].int().clamp(0, 19))
        traffic_light = self.traffic_light_type(inputs[..., 4].int().clamp(0, 3))
        stop_sign = self.stop_sign(inputs[..., 6].unsqueeze(-1))
        sp_limit = self.speed_limit(inputs[..., 7].unsqueeze(-1))

        map_flow_feat = self.map_flow(inputs[..., -3:])
        lane_feat = road_type + traffic_light + stop_sign

        lane_embedding = torch.cat([self_line, map_flow_feat, lane_feat, sp_limit], dim=-1)
    
        # process
        output = self.pointnet(lane_embedding)
        output = self.position_encode(output)

        #max pooling:
        output = torch.max(output, dim=-2).values

        return output

class PredEncoder(nn.Module):
    def __init__(self, dim=256, layers=4, heads=8, dropout=0.1,use_map=True):
        super(PredEncoder, self).__init__()
        self.ego_encoder = AgentEncoder()
        self.neighbor_encoder = AgentEncoder()
        self.use_map =  use_map
        if use_map:
            self.map_encoder = PredLaneEncoder()

        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                         activation='gelu', dropout=dropout, batch_first=True)
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)
        print('vec_encoder',sum([p.numel() for p in self.parameters()]))
    
    def forward(self, inputs):
        ego = inputs['ego_state']
        neighbor = inputs['neighbor_state']

        actors = torch.cat([ego.unsqueeze(1), neighbor], dim=1)
        actors_mask = torch.eq(actors, 0)[:, :, -1, 0]
        actors_mask[:, 0] = False

        ego = self.ego_encoder(ego)
        B, N, T, D = neighbor.shape
        neighbor = self.neighbor_encoder(neighbor.reshape(B*N, T, D))
        neighbor = neighbor.reshape(B, N, -1)

        encode_actors = torch.cat([ego.unsqueeze(1), neighbor], dim=1)
        B,N,C = encode_actors.shape

        if self.use_map:
            maps = inputs['map_segs']
            map_mask = torch.eq(maps, 0)[:, :, 0, 0]
            maps = self.map_encoder(maps)
            encode_actors = torch.cat([encode_actors, maps], dim=1)
            encode_masks = torch.cat([actors_mask, map_mask], dim=1)
        else:
            encode_masks = actors_mask

        encodings = self.fusion_encoder(encode_actors ,src_key_padding_mask=encode_masks)

        encoder_outputs = {
            'actors': actors,
            'encodings': encodings,
            'masks': encode_masks
        }
        
        encoder_outputs.update(inputs)
        return encoder_outputs


from .swin_T import PredSwinTransformerV2


class OGMFlowEncoder(nn.Module):
    def __init__(self,sep_flow=False,large_scale=False):
        super(OGMFlowEncoder, self).__init__()

        self.map_type_encoder = nn.Embedding(num_embeddings=20, embedding_dim=64 ,padding_idx=0)
        self.tl_encoder = nn.Embedding(num_embeddings=4, embedding_dim=64, padding_idx=0)
        self.map_encoder = nn.Linear(1, 64)
        
        self.ogm_rg_encoder = nn.Linear(33 if large_scale else 11, 128)
        self.sep_flow = sep_flow
        if not self.sep_flow:
            self.flow_encoder = nn.Linear(2, 128)
        self.offset_encoder = nn.Linear(2, 64)

        self.point_net = nn.Sequential(nn.Linear((384 if not sep_flow else 256), 192), nn.ReLU(), nn.Linear(192, 96))
        print('flow_encoder',sum([p.numel() for p in self.parameters()]))
    
    def forward(self, inputs, offsets):
        hist_ogm = inputs['hist_ogm']
        hist_flow = inputs['hist_flow']
        road_graph = inputs['road_graph']
 
        rg, road_type, traffic = road_graph[...,0:1], road_graph[..., 1].int(), road_graph[..., 2].int()

        hist_ogm = self.ogm_rg_encoder(hist_ogm)
        if not self.sep_flow:
            hist_flow = self.flow_encoder(hist_flow)

        offsets = self.offset_encoder(offsets)
        maps = self.map_encoder(rg) + self.tl_encoder(traffic.clamp(0, 3)) + self.map_type_encoder(road_type.clamp(0, 19))
        maps = maps

        if self.sep_flow:
            mm_inputs = torch.cat([hist_ogm, maps, offsets], dim=-1)
        else:
            mm_inputs = torch.cat([hist_ogm, hist_flow, maps, offsets], dim=-1)
        mm_inputs = self.point_net(mm_inputs)
        mm_inputs = mm_inputs.permute(0, 3, 1, 2)
        return mm_inputs

class VisualEncoder(PredSwinTransformerV2):
    def __init__(self, 
                config,
                input_resolution=(512, 512),
                embedding_channels=96,
                window_size=8,
                in_channels=96,
                patch_size=4,
                use_checkpoint=False,
                sequential_self_attention=False,
                use_deformable_block=True,
                large_scale=False,
                **kwargs):
        super(VisualEncoder, self).__init__(input_resolution=input_resolution,
                                            window_size=window_size,
                                            in_channels=in_channels,
                                            use_checkpoint=use_checkpoint,
                                            sequential_self_attention=sequential_self_attention,
                                            embedding_channels=embedding_channels,
                                            patch_size=patch_size,
                                            depths=(2, 2, 2),
                                            number_of_heads=(6, 12, 24),
                                            use_deformable_block=use_deformable_block,
                                            large_scale=large_scale,
                                            **kwargs)

        self.mm_encoder = OGMFlowEncoder(sep_flow=True,large_scale=large_scale)                    
        
        self.config = config
        self.input_resolution = input_resolution
        self._make_position_bias_input()

        self.init_crop = input_resolution[0] / patch_size
        print('swin_encoder',sum([p.numel() for p in self.parameters()]))
    
    def half_cropping(self, tensor, stage=0):
        cropped_len = int(self.init_crop / (2 ** stage))
        begin, end = int(cropped_len / 4), int(3 * cropped_len / 4)
        return tensor[:, :, begin:end, begin:end]
    
    def crop_output_list(self, output_list):
        return [self.half_cropping(outputs, i) for i, outputs in enumerate(output_list)]
    
    def _make_position_bias_input(self):
        device = self.stages[0].blocks[0].window_attention.tau.device
        indexes: torch.Tensor = torch.arange(1, self.config.grid_height_cells*2 + 1, device=device)
        widths_indexes = - (indexes - (self.config.sdc_x_in_grid + self.config.grid_height_cells) - 0.5) / self.config.pixels_per_meter
        heights_indexes = - (indexes - (self.config.sdc_y_in_grid + self.config.grid_width_cells) - 0.5) / self.config.pixels_per_meter
        #correspnding (x,y) in dense coordinates 
        coordinates: torch.Tensor = torch.stack(torch.meshgrid([heights_indexes, widths_indexes]), dim=-1)
        self.register_buffer('input_bias', coordinates)
    
    def forward(self, inputs):
        coordinate_bias = self.input_bias #[b, h, w, 2]
        b = inputs['hist_ogm'].shape[0]
        offsets = coordinate_bias.unsqueeze(0).expand(b, -1, -1, -1)
        visual_inputs = self.mm_encoder(inputs, offsets)
        outputs_list, flow = super(VisualEncoder, self).forward(visual_inputs, inputs['hist_flow'].permute(0, 3, 1, 2))
        outputs_list = self.crop_output_list(outputs_list)
        return outputs_list, None