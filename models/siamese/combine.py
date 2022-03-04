from torch import nn 
import torch
from pathlib import Path
import numpy as np
from .image import MVCNN_feat
from .mesh import MeshNet_feat
from .voxel import VoxNet_feat
from .pointcloud import PointNetCls_feat

class FeatureModel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__() 
        self.model_img_feat = MVCNN_feat(n_view=24, pretrained=True, feature_dim = feature_dim)
        self.model_mesh_feat = MeshNet_feat(feature_dim = feature_dim)
        self.model_pt_feat = PointNetCls_feat(feature_dim = feature_dim)
        self.model_vox_feat = VoxNet_feat(feature_dim = feature_dim)

    def forward(self, data, selection):
        img, mesh, pt, vox  = data
        out_img = self.model_img_feat(img) * selection[:, 0:1]
        out_mesh = self.model_mesh_feat(mesh)  * selection[:, 1:2]
        out_pt = self.model_pt_feat(pt)  * selection[:, 2:3]
        out_vox = self.model_vox_feat(vox)  * selection[:, 3:4]
        # feature: B, D 
        # print(out_img.shape, out_mesh.shape, out_pt.shape, out_vox.shape)
        # print(selection)
        global_features = ((out_img + out_mesh + out_pt + out_vox) / selection.sum(1, keepdim=True)).float()
        return global_features, (out_img, out_mesh, out_pt, out_vox)

class ClassifierModel(nn.Module):
    def __init__(self, feature_dim, n_classes):
        super().__init__() 
        dropout = 0.1
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, n_classes)
            # nn.Linear(feature_dim, feature_dim // 2),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(feature_dim // 2, feature_dim // 4),
            # nn.ReLU(),
            # nn.Dropout(p=dropout),
            # nn.Linear(feature_dim // 4, n_classes)
        )

    def forward(self, features):
        return self.classifier(features)

