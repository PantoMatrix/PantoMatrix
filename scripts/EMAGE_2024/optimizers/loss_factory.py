# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2):
        """ Compute the geodesic distance between two rotation matrices.

        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        m1 = m1.reshape(-1, 3, 3)
        m2 = m2.reshape(-1, 3, 3)
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.clamp(cos, min=-1 + 1E-6, max=1-1E-6)

        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction='mean'):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {reduction}')


class BCE_Loss(nn.Module):
    def __init__(self, args=None):
        super(BCE_Loss, self).__init__()
       
    def forward(self, fake_outputs, real_target):
        final_loss = F.cross_entropy(fake_outputs, real_target, reduce="mean")
        return final_loss

class weight_Loss(nn.Module):
    def __init__(self, args=None):
        super(weight_Loss, self).__init__()
    def forward(self, weight_f):
        weight_loss_div = torch.mean(weight_f[:, :, 0]*weight_f[:, :, 1])
        weight_loss_gap = torch.mean(-torch.log(torch.max(weight_f[:, :, 0], dim=1)[0] - torch.min(weight_f[:, :, 0], dim=1)[0]))
        return weight_loss_div, weight_loss_gap    
    

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
        "GeodesicLoss": GeodesicLoss,
        "weight_Loss": weight_Loss,
    }


def get_loss_func(loss_name, **kwargs):    
    loss_func_class = LOSS_FUNC_LUT.get(loss_name)   
    loss_func = loss_func_class(**kwargs)   
    return loss_func


