import torch.nn as nn
import torch.nn.functional as F

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def temporal_upsample(inputs, size=(2, 2), mode='nearest'):
    assert len(inputs.shape) == 5
    b, c, t, h, w = inputs.shape
    inputs = inputs.permute(0, 2, 1, 3, 4).contiguous().reshape(b*t, c, h, w)
    inputs = F.interpolate(input=inputs, size=(2*h, 2*w), mode=mode, align_corners=True)
    inputs = inputs.reshape(b, t, c, 2*h, 2*w).permute(0, 2, 1, 3, 4)
    return inputs

class CrossAttention(nn.Module):
    def __init__(self, dim=384, heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True,)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim*4, dim), nn.Dropout(0.1))
        self.norm_0 = nn.LayerNorm(dim)
        self.norm_1 = nn.LayerNorm(dim)

    def forward(self, query, key, mask):
        output, _ = self.cross_attention(query, key, key, key_padding_mask=mask)
        attention_output = self.norm_0(output + query)
        n_output = self.ffn(attention_output)
        return self.norm_1(n_output + attention_output)