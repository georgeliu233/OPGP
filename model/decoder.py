import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from .aggregator import CrossAttention, temporal_upsample

import math 


class DecodeUpsample(nn.Module):
    def __init__(self, input_dim, kernel, timestep):
        super(DecodeUpsample, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(input_dim, input_dim//2, (1, kernel, kernel), padding='same'), nn.GELU())
        self.residual_conv = nn.Sequential(nn.Conv3d(input_dim//2, input_dim//2, (timestep, 1, 1)), nn.GELU())

    def forward(self, inputs, res):
        #b, t, c, h, w = inputs.shape
        inputs = temporal_upsample(inputs, mode='bilinear')
        inputs = self.conv(inputs) + self.residual_conv(res)
        return inputs
    

class PredFinalDecoder(nn.Module):
    def __init__(self, input_dim, kernel=3, large_scale=False,planning=True, use_flow=False):
        super(PredFinalDecoder, self).__init__()
        '''
        input h,w = 128
        dual deconv for flow and ogms
        ''' 
        self.input_dim = input_dim
        if large_scale:
            self.ogm_conv = nn.Conv3d(input_dim, 4, (1, kernel, kernel), padding='same')
            self.flow_conv = nn.Conv3d(input_dim, 2, (1, kernel, kernel), padding='same')
        else:
            self.ogm_conv = nn.Sequential(nn.Conv3d(input_dim, input_dim//2, (1, kernel, kernel), padding='same'), 
                            nn.GELU(), nn.Upsample(scale_factor=(1, 2, 2)),
                            nn.Conv3d(input_dim//2, 4 if planning else 2, (1, kernel, kernel), padding='same'))

            self.flow_conv = nn.Sequential(nn.Conv3d(input_dim, input_dim//2, (1, kernel, kernel), padding='same'), 
                            nn.GELU(), nn.Upsample(scale_factor=(1, 2, 2)),
                            nn.Conv3d(input_dim//2, 2, (1, kernel, kernel), padding='same'))
    
    def forward(self, inputs):
        ogms = self.ogm_conv(inputs)
        flows = self.flow_conv(inputs)
        return torch.cat([ogms, flows], dim=1)

class STrajNetDecoder(nn.Module):
    def __init__(self, dim=384, heads=8, len_fpn=2, kernel=3, timestep=5, dropout=0.1,
        flow_pred=False, large_scale=False):
        super(STrajNetDecoder, self).__init__()

        self.timestep = timestep
        self.len_fpn = len_fpn
        self.residual_conv = nn.Sequential(nn.Conv3d(dim, dim, (timestep, 1, 1)), nn.GELU())
        self.aggregator = nn.ModuleList([CrossAttention(dim, heads, dropout) for _ in range(timestep)])

        self.actor_layer = nn.Sequential(nn.Linear(256, dim), nn.GELU())

        self.fpn_decoders = nn.ModuleList([
            DecodeUpsample(dim // (2 ** i), kernel, timestep) for i in range(len_fpn)
        ])

        self.upsample = nn.Upsample(scale_factor=(1, 2, 2))
        if flow_pred:
            self.output_conv = PredFinalDecoder(dim // (2 ** len_fpn),large_scale=large_scale)
        else:
            self.output_conv = nn.Conv3d(dim // (2 ** len_fpn), 4, (1, kernel, kernel), padding='same')
    
    def forward(self, output_list, actor, actor_mask):
        # Aggregations:
        enc_output = output_list[-1]
        b, c, h, w = enc_output.shape
        res_output = enc_output.unsqueeze(2).expand(-1, -1, self.timestep, -1, -1)
        enc_output = enc_output.reshape(b, c, h*w).permute(0, 2, 1)
        #[b, t, h*w, c]
        enc_output = enc_output.unsqueeze(1).expand(-1, self.timestep, -1, -1)

        actor = self.actor_layer(actor)
        actor_mask[:, 0] = False
        agg_output =  torch.stack([self.aggregator[i](enc_output[:, i], actor, actor_mask) for i in range(self.timestep)], dim=2)
        agg_output = agg_output.permute(0, 3, 2, 1).reshape(b, -1, self.timestep, h, w)
        decode_output = agg_output + self.residual_conv(res_output)
        # fpn decoding:
        for j in range(self.len_fpn):
            decode_output = self.fpn_decoders[j](decode_output, output_list[-2-j].unsqueeze(2).expand(-1, -1, self.timestep, -1, -1))
        decode_output = self.output_conv(self.upsample(decode_output))

        #[b, t, c, h, w]
        return decode_output



class EgoPlanner(nn.Module):
    def __init__(self, dim=256, use_dynamic=False,timestep=5):
        super(EgoPlanner,self).__init__()
        self.timestep = timestep
        self.out_step = 2
        self.planner = nn.Sequential(nn.Linear(256, 128), nn.ELU(),nn.Dropout(0.1),
                                    nn.Linear(128, timestep*self.out_step *10))
        self.scorer = nn.Sequential(nn.Linear(256, 128), nn.ELU(),nn.Dropout(0.1),
                                    nn.Linear(128, 1))
        self.use_dynamic = use_dynamic
    
    def physical(self, action, last_state):
        d_t = 0.1 
        d_v = action[:, :, :, 0].clamp(-5, 5)
        d_theta = action[:, :, :, 1].clamp(-1, 1)
        
        x_0 = last_state[:, 0]
        y_0 = last_state[:, 1]
        theta_0 = last_state[:, 4]
        v_0 = torch.hypot(last_state[:, 2], last_state[:, 3]) 

        v = v_0.reshape(-1,1,1) + torch.cumsum(d_v * d_t, dim=-1)
        v = torch.clamp(v, min=0)
        theta = theta_0.reshape(-1,1,1) + torch.cumsum(d_theta * d_t, dim=-1)
        theta = torch.fmod(theta, 2*torch.pi)
        x = x_0.reshape(-1,1,1) + torch.cumsum(v * torch.cos(theta) * d_t, dim=-1)
        y = y_0.reshape(-1,1,1) + torch.cumsum(v * torch.sin(theta) * d_t, dim=-1)
        traj = torch.stack([x, y, theta], dim=-1)
        return traj
    
    def forward(self, features, current_state):
        traj = self.planner(features).reshape(-1, 9, self.timestep*10, self.out_step)
        if self.use_dynamic:
            traj = self.physical(traj, current_state)
        score = self.scorer(features)
        return traj, score


class PlanningDecoder(nn.Module):
    def __init__(self, dim=256, heads=8, dropout=0.1, use_dynamic=False, timestep=5):
        super(PlanningDecoder,self).__init__()

        self.region_embed = nn.Parameter(torch.zeros(1, 9, 256), requires_grad=True)
        nn.init.kaiming_uniform_(self.region_embed)

        self.self_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.bev_attention = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.bev_layer = nn.Sequential(nn.Linear(384, dim), nn.GELU())

        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*4, dim), nn.Dropout(0.1))

        self.norm_0 = nn.LayerNorm(dim)
        self.norm_1 = nn.LayerNorm(dim)

        self.planner = EgoPlanner(dim, use_dynamic, timestep)
    
    def forward(self, inputs):
        #encode the poly plan as ref:
        b = inputs['encodings'].shape[0]
        plan_query = self.region_embed.expand(b,-1,-1)
        self_plan_query,_ = self.self_attention(plan_query, plan_query, plan_query)
        #cross attention with bev and map-actors:
        map_actors = inputs['encodings'][:, 0]
        map_actors_mask = inputs['masks'][:, 0]
        map_actors_mask[:,0] = False
        dense_feature,_ = self.cross_attention(self_plan_query, map_actors, map_actors, key_padding_mask=map_actors_mask)
        b, c, h, w = inputs['bev_feature'].shape
        bev_feature = inputs['bev_feature'].reshape(b, c, h*w).permute(0, 2, 1)
        bev_feature = self.bev_layer(bev_feature)
        bev_feature,_ = self.bev_attention(self_plan_query, bev_feature, bev_feature)

        attention_feature = self.norm_0(dense_feature + bev_feature + plan_query)
        output_feature = self.ffn(attention_feature) + attention_feature
        output_feature = self.norm_1(output_feature)

        # output:
        ego_current = inputs['actors'][:, 0, -1, :]
        traj, score = self.planner(output_feature, ego_current)
        return traj, score.squeeze(-1)