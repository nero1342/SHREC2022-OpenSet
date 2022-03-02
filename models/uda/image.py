import torch
import torchvision
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from utils import init_weights


class FeatureNet(nn.Module):
    def __init__(self, pretrained=False):
        super(FeatureNet, self).__init__()
        self.base_model = torchvision.models.resnet50(pretrained=pretrained)
        self.feature_len = 2048
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    def forward(self, x):
        # feature maps
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        return x

class MVCNN_feat(nn.Module):
    def __init__(self, n_view, pretrained=True):
        super(MVCNN_feat, self).__init__()
        self.n_view = n_view
        self.ft_net = FeatureNet(pretrained=pretrained)
        # self.cls_net = nn.Linear(self.ft_net.feature_len, n_class)

    def forward(self, view_batch):
        assert view_batch.size(1) == self.n_view
        view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
        view_fts = self.ft_net(view_batch)
        local_view_fts = view_fts.view(-1, self.n_view, view_fts.size(-1))
        global_view_fts, _ = local_view_fts.max(dim=1)
        return (global_view_fts, local_view_fts)
   
      

class MVCNN_cls(nn.Module):
    def __init__(self, n_class):
        super(MVCNN_cls, self).__init__()
        ft_len = FeatureNet().feature_len
        print("Feat_len = ", ft_len)
        self.cls_net = weightNorm(nn.Linear(ft_len, n_class), name="weight")
        self.cls_net.apply(init_weights)


    def forward(self, view_fts, global_ft=False, local_ft=False):
        global_view_fts, local_view_fts = view_fts
        outputs = self.cls_net(global_view_fts)
        if global_ft and local_ft:
            return outputs, global_view_fts, local_view_fts
        elif global_ft:
            return outputs, global_view_fts
        elif local_ft:
            return outputs, local_view_fts
        else:
            return outputs
