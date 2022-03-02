from pathlib import Path
from random import shuffle
import torch 
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from tqdm import tqdm 

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def cal_acc(loader, netC, netF, out_file, flag=False, threshold=0.1, epsilon=1e-5, n_class = 8):
    start_test = True
    all_label, all_output = [], []
    fts_img, fts_mesh, fts_pt, fts_vox = [], [], [], []

    with torch.no_grad():
        iter_test = iter(loader)
        for _ in tqdm(range(len(loader))):
            inputs_test = iter_test.next()
            img, mesh, pt, vox, _, lbl, _ = inputs_test

            img = img.cuda()
            mesh = [d.cuda() for d in mesh]
            pt = pt.cuda()
            vox = vox.cuda()
            lbl = lbl.cuda()
            data = (img, mesh, pt, vox)

            inputs = data
            labels = lbl
            # inputs = inputs.cuda()

            out, feas = netC(netF(inputs), global_ft=True)

            out_img, out_mesh, out_pt, out_vox = out
            ft_img, ft_mesh, ft_pt, ft_vox = feas
            out_obj = (out_img + out_mesh + out_pt + out_vox)/4

            # _, preds = torch.max(out_obj, 1)
            # all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
            all_label.extend(lbl.squeeze().detach().cpu().numpy().tolist())
            all_output.extend(out_obj.squeeze().detach().cpu().numpy().tolist())

            fts_img.append(ft_img.detach().cpu().numpy())
            fts_mesh.append(ft_mesh.detach().cpu().numpy())
            fts_pt.append(ft_pt.detach().cpu().numpy())
            fts_vox.append(ft_vox.detach().cpu().numpy())
            
            # if start_test:
            #     all_fea = feas.float().cpu()
            #     all_output = outputs.float().cpu()
            #     all_label = labels.float()
            #     start_test = False
            # else:
            #     all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
            #     all_output = torch.cat((all_output, outputs.float().cpu()), 0)
            #     all_label = torch.cat((all_label, labels.float()), 0)

    fts_img = np.concatenate(fts_img, axis=0)
    fts_mesh = np.concatenate(fts_mesh, axis=0)
    fts_pt = np.concatenate(fts_pt, axis=0)
    fts_vox = np.concatenate(fts_vox, axis=0)
    all_fea = np.concatenate((fts_img, fts_mesh, fts_pt, fts_vox), axis=1)

    dist_mat = scipy.spatial.distance.cdist(all_fea, all_fea, "cosine")
    map_s = map_score(dist_mat, all_label, all_label)

    all_label = torch.FloatTensor(all_label)
    all_output = torch.FloatTensor(all_output)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        # all_output = nn.Softmax(dim=1)(all_output)
        ent = torch.sum(-all_output * torch.log(all_output + epsilon), dim=1) / np.log(n_class)

        from sklearn.cluster import KMeans
        kmeans = KMeans(2, random_state=0).fit(ent.reshape(-1,1))
        labels = kmeans.predict(ent.reshape(-1,1))

        idx = np.where(labels==1)[0]
        iidx = 0
        if ent[idx].mean() > ent.mean():
            iidx = 1
        predict[np.where(labels==iidx)[0]] = n_class

        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        # import pdb; pdb.set_trace()
        for each in matrix:
            out_file.write(str(each.tolist()) + '\n')
            out_file.flush()

        out_file.write('\n\n')
        out_file.flush()

        matrix = matrix[np.unique(all_label).astype(int),:]
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        unknown_acc = acc[-1:].item()
        # acc = 100
        # unknown_acc = 100
        return np.mean(acc[:-1]), np.mean(acc), unknown_acc, map_s
        # return acc, acc, unknown_acc, map_s
    else:
        return accuracy*100, mean_ent

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            param.grad.data.clamp_(-grad_clip, grad_clip)
            
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss

def split_trainval(data_root: str, train_ratio=0.8):
    train_root = Path(data_root) / "train"
    cates = sorted([d.stem for d in train_root.glob('*') if d.is_dir()])
    train_list, val_list = [], []
    for idx, cate in enumerate(cates):
        samples = [d for d in (train_root / cate).glob('*') if d.is_dir()]
        shuffle(samples)
        len_train = int(len(samples) * train_ratio)
        assert len_train > 0
        for _i, sample in enumerate(samples):
            if _i < len_train:
                train_list.append({'path': str(sample.absolute()),'label': idx})
            else:
                val_list.append({'path': str(sample.absolute()),'label': idx})
    return train_list, val_list


def res2tab(res: dict, n_palce=4):
    def dy_str(s, l):
        return  str(s) + ' '*(l-len(str(s)))
    min_size = 8
    k_str, v_str = '', ''
    for k, v in res.items():
        if (k == "epoch"):
            break
        cur_len = max(min_size, len(k)+2)
        k_str += dy_str(f'{k}', cur_len) + '| '
        v_str += dy_str(f'{v:.4}', cur_len) + '| '
    return k_str, v_str

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


################################### metric #######################################
import scipy
import scipy.spatial
import numpy as np


def acc_score(y_true, y_pred, average="micro"):
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if average == "micro": 
        # overall
        return np.mean(y_true == y_pred)
    elif average == "macro":
        # average of each class
        cls_acc = []
        for cls_idx in np.unique(y_true):
            cls_acc.append(np.mean(y_pred[y_true==cls_idx]==cls_idx))
        return np.mean(np.array(cls_acc))
    else:
        raise NotImplementedError


def map_score(dist_mat, lbl_a, lbl_b, metric='cosine'):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p = 0.0
        r = 0.0
        for j in range(n_b):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res.append(p/r)
        else:
            res.append(0)
    return np.mean(res)


def map_score(dist_mat, lbl_a, lbl_b):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        p = 0.0
        r = 0.0
        for j in range(n_b):
            if lbl_a[i] == lbl_b[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res.append(p/r)
        else:
            res.append(0)
    return np.mean(res)


def nn_score(dist_mat, lbl_a, lbl_b):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        if lbl_a[i] == lbl_b[order[0]]:
            res.append(1)
        else:
            res.append(0)
    return np.mean(res)


def ndcg_score(dist_mat, lbl_a, lbl_b, k=100):
    n_a, n_b = dist_mat.shape
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        order = s_idx[i]
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, n_b + 2)))
        dcg = np.cumsum([1.0/np.log2(idx+2) if lbl_a[i] == lbl_b[item] else 0.0 for idx, item in enumerate(order)])
        ndcg = (dcg/idcg)[k-1]
        res.append(ndcg)
    return np.mean(res)


def anmrr_score(dist_mat, lbl_a, lbl_b):
    # NG: number of ground truth images (target images) per query (vector)
    n_a, n_b = dist_mat.shape
    lbl_a, lbl_b = np.array(lbl_a), np.array(lbl_b)
    NG = np.array([(lbl_a[i]==lbl_b).sum() for i in range(lbl_a.shape[0])])
    s_idx = dist_mat.argsort()
    res = []
    for i in range(n_a):
        cur_NG = NG[i]
        K = min(4*cur_NG, 2*NG.max())
        order = s_idx[i]
        ARR = np.sum([(idx+1)/cur_NG if lbl_a[i] == lbl_b[order[idx]] else (K+1)/cur_NG for idx in range(cur_NG)])
        MRR = ARR - 0.5*cur_NG - 0.5
        NMRR = MRR / (K - 0.5*cur_NG + 0.5)
        res.append(NMRR)
    return np.mean(res)


if __name__ == "__main__":
    split_trainval('data/OS-MN40')
