import os
import shutil
import time
import json

# from pyrsistent import T
import torch
import random
import numpy as np
import pickle
from tqdm import tqdm
import scipy.spatial
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from models.uda_efficient import UniModel_cls, UniModel_base

from loaders.source import OSMN40_retrive
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

######### must config this #########
data_root = Path('../data/OS-MN40')
typedata = "full"

ckpt_path = "./cache/ckpts_target_test/OS-MN40_2022-03-02-08-43-58"
ckptF_path = ckpt_path + '/ckpt_target_F.pth'
ckptC_path = ckpt_path + '/ckpt_target_C.pth'
save_path = './pickle/target_test/'
####################################

# configure
dist_mat_path = Path(ckpt_path)/ "cdist_cosine.txt"
dist_metric= 'cosine'
# dist_metric = 'euclidean'
batch_size = 8
n_worker = 4
n_class = 8

save_vec = False
if (os.path.isdir(save_path) == False):
    save_vec = True   
    os.mkdir(save_path)  


def extract(query_loader, target_loader, netF, netC):
    netF.eval()
    netC.eval()
    print("Extracting....")

    q_fts_img, q_fts_mesh, q_fts_pt, q_fts_vox = [], [], [], []
    t_fts_img, t_fts_mesh, t_fts_pt, t_fts_vox = [], [], [], []

    st = time.time()

    if (save_vec):
        for img, mesh, pt, vox, _ in tqdm(query_loader):
            img = img.cuda()
            mesh = [d.cuda() for d in mesh]
            pt = pt.cuda()
            vox = vox.cuda()
            data = (img, mesh, pt, vox)
            _, ft = netC(netF(data), global_ft=True)
            ft_img, ft_mesh, ft_pt, ft_vox = ft
            q_fts_img.append(ft_img.detach().cpu().numpy())
            q_fts_mesh.append(ft_mesh.detach().cpu().numpy())
            q_fts_pt.append(ft_pt.detach().cpu().numpy())
            q_fts_vox.append(ft_vox.detach().cpu().numpy())
        q_fts_img = np.concatenate(q_fts_img, axis=0)
        q_fts_mesh = np.concatenate(q_fts_mesh, axis=0)
        q_fts_pt = np.concatenate(q_fts_pt, axis=0)
        q_fts_vox = np.concatenate(q_fts_vox, axis=0)
        q_fts_uni = np.concatenate((q_fts_img, q_fts_mesh, q_fts_pt, q_fts_vox), axis=1)
        with open(save_path + 'query.pickle', 'wb') as handle:
            pickle.dump(q_fts_uni, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path + 'query.pickle', 'rb') as handle:
            q_fts_uni = pickle.load(handle)

    if (save_vec):
        for img, mesh, pt, vox, _ in tqdm(target_loader):
            img = img.cuda()
            mesh = [d.cuda() for d in mesh]
            pt = pt.cuda()
            vox = vox.cuda()
            data = (img, mesh, pt, vox)
            _, ft = netC(netF(data), global_ft=True)
            ft_img, ft_mesh, ft_pt, ft_vox = ft
            t_fts_img.append(ft_img.detach().cpu().numpy())
            t_fts_mesh.append(ft_mesh.detach().cpu().numpy())
            t_fts_pt.append(ft_pt.detach().cpu().numpy())
            t_fts_vox.append(ft_vox.detach().cpu().numpy())
        t_fts_img = np.concatenate(t_fts_img, axis=0)
        t_fts_mesh = np.concatenate(t_fts_mesh, axis=0)
        t_fts_pt = np.concatenate(t_fts_pt, axis=0)
        t_fts_vox = np.concatenate(t_fts_vox, axis=0)
        t_fts_uni = np.concatenate((t_fts_img, t_fts_mesh, t_fts_pt, t_fts_vox), axis=1)
        with open(save_path + 'target.pickle', 'wb') as handle:
            pickle.dump(t_fts_uni, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path + 'target.pickle', 'rb') as handle:
            t_fts_uni = pickle.load(handle)

    print(f"Time Cost: {time.time()-st:.4f}")

    dist_mat = scipy.spatial.distance.cdist(q_fts_uni, t_fts_uni, dist_metric)
    np.savetxt(str(dist_mat_path), dist_mat)

def read_object_list(filename, pre_path):
    object_list = []
    with open(filename, 'r') as fp:
        for name in fp.readlines():
            if name.strip():
                object_list.append(str(pre_path/name.strip()))
    return object_list


def main():
    # init train_loader and test loader
    print("Loader Initializing...\n")
    query_list = read_object_list("query.txt", data_root / "query")
    target_list = read_object_list("target.txt", data_root / "target")
    query_data = OSMN40_retrive(query_list, typedata= typedata)
    target_data = OSMN40_retrive(target_list, typedata= typedata)
    print(f'query samples: {len(query_data)}')
    print(f'target samples: {len(target_data)}')
    query_loader = DataLoader(query_data, batch_size=batch_size, shuffle=False,
                                               num_workers=n_worker)
    target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=False,
                                             num_workers=n_worker)
    print(f"Loading model from {ckpt_path}")

    netF = UniModel_base(n_class)
    netC = UniModel_cls(n_class)
    # import pdb; pdb.set_trace()
    ckpt = torch.load(ckptF_path, map_location=torch.device('cpu'))
    netF.load_state_dict(ckpt['net'])
    netF = netF.cuda()
    netF = nn.DataParallel(netF)

    ckpt = torch.load(ckptC_path, map_location=torch.device('cpu'))
    netC.load_state_dict(ckpt['net'])
    netC = netC.cuda()
    netC = nn.DataParallel(netC)

    # extracting
    with torch.no_grad():
        extract(query_loader, target_loader, netF, netC)

    print(f"cdis matrix can be find in path: {dist_mat_path.absolute()}")


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
