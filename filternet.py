from fastai.column_data import emb_init
from fastai.core import A, V
from fastai.dataloader import DataLoader
from fastai.dataset import ModelData
from fastai.layers import Flatten
import numpy as np
import torch
from torch import nn
from torch.nn.init import kaiming_normal
from torch.utils.data import Dataset
import torch.nn.functional as F


def conv_layer(window, ks=3, dilation=1):
    return nn.Sequential(
        nn.Conv1d(1, 1, kernel_size=ks, bias=False, dilation=dilation),
        nn.AdaptiveAvgPool1d(window),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))


class FilterNet24H2(nn.Module):
    def __init__(self, emb_szs, n_cont, emb_drop, out_sz, szs, drops, window=24, filters=[1, 2, 3, 4, 5, 6],
                 y_range=None, use_bn=False, is_reg=True, is_multi=False):
        super().__init__()

        # TODO: Use the filters arg to generate the conv_layers dynamically
        # Wavenet model layers
        self.c1a = conv_layer(window=window // 2, ks=1, dilation=1)
        self.c1b = conv_layer(window=window // 4, ks=1, dilation=2)
        self.c2a = conv_layer(window=window // 2, ks=2, dilation=1)
        self.c2b = conv_layer(window=window // 4, ks=2, dilation=2)
        self.c3a = conv_layer(window=window // 2, ks=3, dilation=1)
        self.c3b = conv_layer(window=window // 4, ks=3, dilation=2)
        self.c4a = conv_layer(window=window // 2, ks=4, dilation=1)
        self.c4b = conv_layer(window=window // 4, ks=4, dilation=2)
        self.c5a = conv_layer(window=window // 2, ks=5, dilation=1)
        self.c5b = conv_layer(window=window // 4, ks=5, dilation=2)
        self.c6a = conv_layer(window=window // 2, ks=6, dilation=1)
        self.c6b = conv_layer(window=window // 4, ks=6, dilation=2)

        num_wave_outputs = (len(filters) * (window // 2)) + (len(filters) * (window // 4))

        # Fastai's Mixed Input model
        self.embs = nn.ModuleList([nn.Embedding(c, s) for c, s in emb_szs])
        for emb in self.embs: emb_init(emb)
        n_emb = sum(e.embedding_dim for e in self.embs)
        self.n_emb, self.n_cont = n_emb, n_cont

        szs = [n_emb + n_cont] + szs
        self.lins = nn.ModuleList([
            nn.Linear(szs[i], szs[i + 1]) for i in range(len(szs) - 1)])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: kaiming_normal(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        kaiming_normal(self.outp.weight.data)

        self.emb_drop = nn.Dropout(emb_drop)
        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn, self.y_range = use_bn, y_range
        self.is_reg = is_reg
        self.is_multi = is_multi

        # Final layer
        self.f = Flatten()
        self.lin = nn.Linear(szs[-1] + num_wave_outputs, out_sz, bias=False)

    def forward(self, x_window, x_cat, x_cont):
        # TODO: Use the filters arg to generate the conv_layers dynamically
        # Wavenet model
        self.f1a = self.c1a(x_window)
        self.f1b = self.c1b(self.f1a)
        self.f2a = self.c2a(x_window)
        self.f2b = self.c2b(self.f2a)
        self.f3a = self.c3a(x_window)
        self.f3b = self.c3b(self.f3a)
        self.f4a = self.c4a(x_window)
        self.f4b = self.c4b(self.f4a)
        self.f5a = self.c5a(x_window)
        self.f5b = self.c5b(self.f5a)
        self.f6a = self.c6a(x_window)
        self.f6b = self.c6b(self.f6a)
        self.ffc = torch.cat([self.f1a, self.f1b, self.f2a, self.f2b,
                              self.f3a, self.f3b, self.f4a, self.f4b,
                              self.f5a, self.f5b, self.f6a, self.f6b, ], 2)

        # Fastai's Mixed Input Model
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            x2 = self.bn(x_cont) if self.use_bn else x_cont
            x = torch.cat([x, x2], 1) if self.n_emb != 0 else x2
        for l, d, b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: x = b(x)
            x = d(x)

        # Combine results from both nets
        x = x.unsqueeze(1)
        self.fc = torch.cat([self.ffc, x], 2)
        self.flin = self.lin(self.f(self.fc))
        return self.flin


class FilterNetDataset(Dataset):
    def __init__(self, x_window, x_cat, x_cont, y):
        self.x_window = x_window
        self.x_cat = x_cat
        self.x_cont = x_cont
        self.y = y

    def __getitem__(self, idx): return A(self.x_window[idx], self.x_cat[idx], self.x_cont[idx], self.y[idx])
    def __len__(self): return max(len(self.x_window), len(self.x_cat), len(self.x_cont))

