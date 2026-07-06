import torch
import torch.nn as nn
from functional import pcc, jsd

LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator

# Mean Squared Error Loss
@register_loss("MSE")
def build_mse_loss(config):
    return nn.MSELoss()

class PCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        cc = pcc(pred, gt)
        return torch.mean(1 - cc) # loss to minimize

# Pearson Correlation Coefficient Loss
@register_loss("PCC")
def build_pcc_loss(config):
    return PCCLoss()

class JSSLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        return jsd(pred, gt) # since JSD is a divergence, minimizing it acts as a loss (JSS = 1 - JSD)

# Jensen-Shannon Divergence Loss
@register_loss("JSS")
def build_jss_loss(config):
    return JSSLoss()

def build_loss(config):
    """Factory function called by train.py"""
    name = config["name"]
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss {name} not found in registry.")
    return LOSS_REGISTRY[name](config)
