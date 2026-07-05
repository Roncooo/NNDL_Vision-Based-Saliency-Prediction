import torch.nn as nn

LOSS_REGISTRY = {}

def register_loss(name):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator

@register_loss("MSE")
def build_mse_loss(config):
    return nn.MSELoss()

@register_loss("KLDiv")
def build_kl_loss(config):
    return nn.KLDivLoss(reduction='batchmean')

def build_loss(config):
    """Factory function called by train.py"""
    name = config["name"]
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Loss {name} not found in registry.")
    return LOSS_REGISTRY[name](config)
