import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import torch.nn as nn

from .encoder import *
from .decoder import *

class InteractPlanner(nn.Module):
    def __init__(self, config, dim=256, enc_layer=2, heads=8, dropout=0.1,
        timestep=5, decoder_dim=384, fpn_len=2, use_dynamic=True,
        large_scale=True,flow_pred=False):

        super(InteractPlanner, self).__init__()

        self.visual_encoder = VisualEncoder(config, input_resolution=(256, 256), 
                                                 patch_size=2,use_deformable_block=False,large_scale=large_scale)
        
        self.vector_encoder = VectorEncoder(dim, enc_layer, heads, dropout)

        self.ogm_decoder = STrajNetDecoder(decoder_dim, heads, len_fpn=fpn_len, timestep=timestep, dropout=dropout,
        flow_pred=flow_pred,large_scale=large_scale)
      
        self.plan_decoder = PlanningDecoder(dim, heads, dropout, use_dynamic=use_dynamic, 
                                            timestep=timestep)
    
    
    def forward(self, inputs):
        encoder_outputs = self.vector_encoder(inputs)
        bev_list, _ = self.visual_encoder(inputs)
        encoder_outputs.update({
            'bev_feature' : bev_list[-1]
        })
        bev_pred = self.ogm_decoder(bev_list, encoder_outputs['encodings'][:, 0], encoder_outputs['masks'][:, 0])
        traj, score = self.plan_decoder(encoder_outputs)
        return bev_pred, traj, score