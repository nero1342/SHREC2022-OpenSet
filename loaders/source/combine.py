import torch
from pathlib import Path
from torch.utils.data import Dataset
from .image import load_img
from .voxel import load_vox
from .mesh import load_mesh
from loaders.source import load_pt
import numpy as np


class OSMN40_train(Dataset):
    def __init__(self, phase, object_list, typedata="full"):
        super().__init__()
        assert phase in ('train', 'val', 'target', 'test')
        self.phase = phase
        self.object_list = object_list
        self.typedata = typedata

    def __getitem__(self, index):
        p = Path(self.object_list[index]['path'])
        lbl = self.object_list[index]['label']
        num_obj = torch.tensor([1, 1, 1, 1])
        # # image
        try:
          img = load_img(p/'image', self.phase in ['train', 'target'], n_view=24)
        except:
          num_obj[0] = 0
          img = torch.zeros((24, 3, 224, 224))
        # # mesh
        try:
          mesh = load_mesh(p/'mesh', self.phase in ['train', 'target'], typedata=self.typedata)
        except:
          mesh = (torch.zeros((15, 500)), torch.zeros((500, 3)))
          num_obj[1] = 0
        # point cloud
        try:
          pt = load_pt(p/'pointcloud',self.phase in ['train', 'target'], resolution=2048)
        except:
          num_obj[2] = 0
          pt = torch.zeros((2048, 3))
        # voxel
        try:
          vox = load_vox(p/'voxel', self.phase in ['train', 'target'], resolution=64)
        except:
          num_obj[3] = 0
          vox = torch.zeros((1, 64, 64, 64))
        print(index, self.object_list[index]['path'], num_obj)
        
        return img, mesh, pt, vox, num_obj, lbl

    def __len__(self):
        return len(self.object_list)


class OSMN40_retrive(Dataset):
    def __init__(self, object_list, typedata="full"):
        super().__init__()
        self.object_list = object_list
        self.typedata = typedata

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        
        p = Path(self.object_list[index])
        # # image
        img = load_img(p/'image', n_view=24)
        # # mesh
        mesh = load_mesh(p/'mesh', typedata = self.typedata)
        # point cloud
        pt = load_pt(p/'pointcloud' , resolution=2048)
        # voxel
        vox = load_vox(p/'voxel', resolution=64)

        if (self.typedata != "full"):
            num_obj = np.loadtxt(p/'mask.txt')
        else:
            num_obj = 1

        return img, mesh, pt, vox, num_obj

    def __len__(self):
        return len(self.object_list)


if __name__ == '__main__':
    pass

