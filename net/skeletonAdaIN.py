import torch
import torch.nn as nn
import numpy as np
import random
import math

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()#N,C,T,V
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
def calc_mean_std2(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    #only V dim
    size = feat.size()#N,C,T,V
    assert (len(size) == 4)
    N, C, T = size[:3]
    feat_var = feat.view(N, C, T, -1).var(dim=3) + eps
    feat_std = feat_var.sqrt().view(N, C, T, 1)
    feat_mean = feat.view(N, C, T, -1).mean(dim=3).view(N, C, T, 1)
    return feat_mean, feat_std
def AdaIN(content_feat, style_feat, dim='TV'):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    N,C,T,V,M = content_feat.shape
    content_feat = content_feat.permute(0,4,1,2,3)
    content_feat = content_feat.reshape(N * M, C, T, V)
    style_feat = style_feat.permute(0,4,1,2,3)
    style_feat = style_feat.reshape(N * M, C, T, V)    

    size = content_feat.size()
    if dim=='TV':
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)
    elif dim=='V':
        style_mean, style_std = calc_mean_std2(style_feat)
        content_mean, content_std = calc_mean_std2(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    ans = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    ans = ans.reshape(N,M,C,T,V)
    ans = ans.permute(0,2,3,4,1)# NCTVM
    return ans 