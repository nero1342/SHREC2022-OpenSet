import torch
import torchvision
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
from efficientnet_pytorch import EfficientNet


class FeatureNet(nn.Module):
    def __init__(self, pretrained=False):
        super(FeatureNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        self.feature_len = 1280 * 7 * 7
        # self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    def forward(self, x):
        # feature maps
        x = self.base_model.extract_features(x)
        # print(x.shape)
        # import pdb; pdb.set_trace
        # flatten
        x = x.view(x.size(0), -1)
        return x

class MVCNN_feat(nn.Module):
    def __init__(self, feature_dim, n_view, pretrained=True):
        super(MVCNN_feat, self).__init__()
        self.n_view = n_view
        self.ft_net = FeatureNet(pretrained=pretrained)
        self.cls_net = nn.Linear(self.ft_net.feature_len, feature_dim)

    def forward(self, view_batch):
        assert view_batch.size(1) == self.n_view
        view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
        view_fts = self.ft_net(view_batch)
        local_view_fts = view_fts.view(-1, self.n_view, view_fts.size(-1))
        global_view_fts, _ = local_view_fts.max(dim=1)
        out = self.cls_net(global_view_fts)
        return out
         