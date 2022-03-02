from re import T
import torch
import torch.nn as nn
from .image import MVCNN_feat, MVCNN_cls
from .mesh import MeshNet_feat, MeshNet_cls
from .voxel import VoxNet_feat, VoxNet_cls
from .pointcloud import PointNetCls_feat, PointNetCls_cls
from .pointmlp import PointMLP


class UniModel_base(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_img_feat = MVCNN_feat(n_view=24, pretrained=True)
        self.model_mesh_feat = MeshNet_feat(n_class)
        self.model_pt_feat = PointNetCls_feat()
        self.model_vox_feat = VoxNet_feat()

    def forward(self, data):
        img, mesh, pt, vox  = data
        out_img = self.model_img_feat(img)
        out_mesh = self.model_mesh_feat(mesh)
        out_pt = self.model_pt_feat(pt)
        out_vox = self.model_vox_feat(vox)
        return (out_img, out_mesh, out_pt, out_vox)
            
class UniModel_cls(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.model_img_cls = MVCNN_cls(n_class)
        self.model_mesh_cls = MeshNet_cls(n_class)
        self.model_pt_cls = PointNetCls_cls(n_class)
        self.model_vox_cls = VoxNet_cls(n_class)

    def forward(self, data, global_ft=False):
        img, mesh, pt, vox  = data
        if global_ft:
            out_img, ft_img = self.model_img_cls(img, global_ft)
            out_mesh, ft_mesh = self.model_mesh_cls(mesh, global_ft)
            out_pt, ft_pt = self.model_pt_cls(pt, global_ft)
            out_vox, ft_vox = self.model_vox_cls(vox, global_ft)
            return (out_img, out_mesh, out_pt, out_vox), (ft_img, ft_mesh, ft_pt, ft_vox)
        else:
            out_img = self.model_img_cls(img)
            out_mesh = self.model_mesh_cls(mesh)
            out_pt = self.model_pt_cls(pt)
            out_vox = self.model_vox_cls(vox)
            return (out_img, out_mesh, out_pt, out_vox)


if __name__ == "__main__":
    pass
