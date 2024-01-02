# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BCE_Loss(nn.Module):
    def __init__(self, args=None):
        super(BCE_Loss, self).__init__()
       
    def forward(self, fake_outputs, real_target):
        final_loss = F.cross_entropy(fake_outputs, real_target, reduce="mean")
        return final_loss


class HuberLoss(nn.Module):
    def __init__(self, beta=0.1, reduction="mean"):
        super(HuberLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss(outputs / self.beta, targets / self.beta, reduction=self.reduction) * self.beta
        return final_loss
    

class KLDLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(KLDLoss, self).__init__()
        self.beta = beta
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss((outputs / self.beta, targets / self.beta) * self.beta)
        return final_loss


class REGLoss(nn.Module):
    def __init__(self, beta=0.1):
        super(REGLoss, self).__init__()
        self.beta = beta
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss((outputs / self.beta, targets / self.beta) * self.beta)
        return final_loss    


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
    
    def forward(self, outputs, targets):
        final_loss = F.l2_loss(outputs, targets)
        return final_loss    

LOSS_FUNC_LUT = {
        "bce_loss": BCE_Loss,
        "l2_loss": L2Loss,
        "huber_loss": HuberLoss,
        "kl_loss": KLDLoss,
        "id_loss": REGLoss,
    }


def get_loss_func(loss_name, **kwargs):    
    loss_func_class = LOSS_FUNC_LUT.get(loss_name)   
    loss_func = loss_func_class(**kwargs)   
    return loss_func


