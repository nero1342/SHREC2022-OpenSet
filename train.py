import os
import time
import json
import torch
import random
import numpy as np
import scipy.spatial
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import custom_loss as custom_loss
from utils import clip_gradient
from tqdm import tqdm 

from models.siamese import FeatureModel, ClassifierModel
from models.uda_efficient import UniModel_cls, UniModel_base
from loaders.source import OSMN40_train
from utils import split_trainval, AverageMeter, res2tab, acc_score, map_score, op_copy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

######### must config this #########
data_root = '/home/pbdang/Contest/SHREC22/OpenSet/data/OS-MN40-Miss1'
typedata = "miss"
####################################

# configure
n_class = 8
n_worker = 4
max_epoch = 150
batch_size = 4
learning_rate = 0.01
this_task = f"OS-MN40_{time.strftime('%Y-%m-%d-%H-%M-%S')}"

# log and checkpoint
out_dir = Path('cache')
save_dir = out_dir/'ckpts_source_b2'/this_task
save_dir.mkdir(parents=True, exist_ok=True)

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def setup_seed():
    seed = 42 # time.time() % 1000_000
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")


def train(data_loader, netC, netF, criterion, optimizer, epoch, iter_num ,max_iter):
    print(f"Epoch {epoch}, Training...")

    optim_cls, optim_centers = optimizer
    crt_cls, crt_tlc, w1, w2 = criterion
    # netC, netF = model
 
    netF.train()
    netC.train()
    loss_meter = AverageMeter()
    tpl_losses = AverageMeter()

    all_lbls, all_preds = [], []

    st = time.time()
    progressbar = tqdm(enumerate(data_loader))
    for i, (img, mesh, pt, vox, selection, lbl) in progressbar:
        # print(selection, selection.shape)
        iter_num += 1
        img = img.cuda()
        mesh = [d.cuda() for d in mesh]
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        selection = selection.cuda()
        data = (img, mesh, pt, vox)

        global_feature, features = netF(data, selection)
        out_obj = netC(global_feature)
        
        cls_loss = crt_cls(out_obj, lbl)
        tpl_loss, _ = crt_tlc(out_obj, lbl)

        loss = w1 * cls_loss + w2 * tpl_loss

        lr_scheduler(optim_cls, iter_num=iter_num, max_iter=max_iter)

        optim_cls.zero_grad()
        optim_centers.zero_grad()

        loss.backward()
        clip_gradient(optim_centers, 0.05)

        optim_cls.step()
        optim_centers.step()


        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        loss_meter.update(loss.item(), lbl.shape[0])
        try:
            tpl_losses.update(tpl_loss.item(), lbl.shape[0])
        except:
            tpl_losses.update(tpl_loss, lbl.shape[0])
        progressbar.set_description(f"\t[{i}/{len(data_loader)}], Loss {loss.item():.4f}")
        # print(f"\t[{i}/{len(data_loader)}], Loss {loss.item():.4f}")

    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s, Loss: {loss_meter.avg:4f}, Tpl_Loss: {tpl_losses.avg:4f}")
    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")
    return iter_num


def validation(data_loader, netC, netF, epoch):
    print(f"Epoch {epoch}, Validation...")

    # netC, netF = model
    netF.eval()
    netC.eval()
    all_lbls, all_preds = [], []
    fts = []

    st = time.time()
    for img, mesh, pt, vox, selection, lbl in tqdm(data_loader):
        img = img.cuda()
        mesh = [d.cuda() for d in mesh]
        pt = pt.cuda()
        vox = vox.cuda()
        lbl = lbl.cuda()
        data = (img, mesh, pt, vox)
        selection = selection.cuda()

        glo_ft, features = netF(data, selection)
        out_obj = netC(glo_ft)
    
        _, preds = torch.max(out_obj, 1)
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
        fts.append(glo_ft.detach().cpu().numpy())

    fts_uni = np.concatenate((fts), axis=0)
    dist_mat = scipy.spatial.distance.cdist(fts_uni, fts_uni, "cosine")
    map_s = map_score(dist_mat, all_lbls, all_lbls)
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Epoch: {epoch}, Time: {time.time()-st:.4f}s")

    res = {
        "overall acc": acc_mi,
        "meanclass acc": acc_ma,
        "map": map_s,
        "epoch": epoch,
        "name": "model: EfficientNetB0"
    }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Epoch Done!\n")
    return map_s, res


def save_checkpoint(val_state, res, name_model, net: nn.Module):
    state_dict = net.state_dict()
    ckpt = dict(
        val_state=val_state,
        res=res,
        net=state_dict,
    )
    fname = 'ckpt_' + name_model + '.pth'
    torch.save(ckpt, str(save_dir / fname))
    fname = 'ckpt_' + name_model + '.meta'
    with open(str(save_dir / fname), 'w') as fp:
        json.dump(res, fp)


def main():
    setup_seed()
    # init train_loader and val_loader
    print("Loader Initializing...\n")
    # import pdb; pdb.set_trace()
    train_list, val_list = split_trainval(data_root)
    # import pdb; pdb.set_trace()
    train_data = OSMN40_train('train', train_list,typedata=typedata)
    val_data = OSMN40_train('val', val_list,typedata=typedata)
    print(f'train samples: {len(train_data)}')
    print(f'val samples: {len(val_data)}')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=n_worker, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                                num_workers=n_worker, drop_last=False)
   
    ## MODEL 
    print("Create new model")
    netF = FeatureModel(feature_dim=256).cuda()
    netC = ClassifierModel(feature_dim=256, n_classes=8).cuda()
    netF = nn.DataParallel(netF)
    netC = nn.DataParallel(netC)
    
    # # model = (netC, netF)

    ### LOSS FUNCTION
    # classification loss 
    crt_cls = nn.CrossEntropyLoss().cuda()
    # triplet center loss 
    crt_tlc = custom_loss.TripletCenterLoss(num_classes=n_class).cuda()
    crt_tlc = torch.nn.utils.weight_norm(crt_tlc, name='centers')
    criterion = [crt_cls, crt_tlc, 1, 0.2]

    ### OPTIMIZER
    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
        
    optim_cls = optim.SGD(param_group)
    optim_cls = op_copy(optim_cls)
    optim_centers = optim.SGD(crt_tlc.parameters(), lr=0.1)
    optimizer = (optim_cls, optim_centers)


    best_res, best_state = None, 0

    max_iter = max_epoch * len(train_loader)
    # interval_iter = max_iter // 10
    iter_num = 0

    for epoch in range(max_epoch):
        # train
        iter_num = train(train_loader, netC, netF, criterion, optimizer, epoch, iter_num, max_iter)
        
        # lr_scheduler.step()
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_state, res = validation(val_loader, netC, netF, epoch)
            # save checkpoint
            if val_state > best_state:
                print("saving model...")
                best_res, best_state = res, val_state
                save_checkpoint(val_state, res, 'F', netF.module)
                save_checkpoint(val_state, res, 'C', netC.module)

            save_checkpoint(val_state, res, 'lastF', netF.module)
            save_checkpoint(val_state, res, 'lastC', netC.module)

    print("\nTrain Finished!")
    tab_head, tab_data = res2tab(best_res)
    print(tab_head)
    print(tab_data)
    print(f'checkpoint can be found in {save_dir}!')
    return best_res


if __name__ == '__main__':
    all_st = time.time()
    main()
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
