import torch
import torch.nn as nn
from functional import pcc, jsd, kl

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

class CombinedLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Retrieve alpha from config, default to 0.5 if not provided
        self.alpha = config.get("alpha", 0.5)

    def forward(self, pred, gt):
        kl_val = kl(pred, gt)
        pcc_val = torch.mean(pcc(pred, gt))
        return kl_val - self.alpha * pcc_val

# Combined Loss: KL - alpha * PCC
@register_loss("combined")
def build_combined_loss(config):
    return CombinedLoss(config)

def build_loss(config):
    """Factory function called by train.py"""
    name = config["name"]
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss {name} not found in registry.")
    return LOSS_REGISTRY[name](config)
