#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: voxnet.py
Created: 2020-01-21 21:32:40
Author : Yangmaonan
Email : 59786677@qq.com
Description: VoxNet 网络结构
'''

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.utils.weight_norm as weightNorm
# from utils import init_weights


class VoxNet_feat(nn.Module):
    def __init__(self, feature_dim):
        super(VoxNet_feat, self).__init__()
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3)),
            
        ]))
        self.pre_mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(32 * 14 * 14 * 14, feature_dim)),
            # ('relu1', torch.nn.ReLU()),
            # ('drop3', torch.nn.Dropout(p=0.4))
        ]))

    def forward(self, x):
        x = self.feat(x)
        g_ft = x.view(x.size(0), -1)
        g_ft = self.pre_mlp(g_ft)
        return g_ft

class VoxNet_cls(nn.Module):
    def __init__(self, n_classes):
        super(VoxNet_cls, self).__init__()
        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc2', weightNorm(torch.nn.Linear(256, n_classes), name="weight"))
        ]))

        self.mlp.apply(init_weights)

    def forward(self, g_ft, global_ft=False):
        x = self.mlp(g_ft)
        if global_ft:
            return x, g_ft
        else:
            return x


if __name__ == "__main__":
    voxnet = VoxNet(32, 10)
    data = torch.rand([256, 1, 32, 32, 32])
    voxnet(data)
