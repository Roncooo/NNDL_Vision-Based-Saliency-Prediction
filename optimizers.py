import torch

OPTIMIZER_REGISTRY = {}

def register_optimizer(name):
    def decorator(cls):
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator

@register_optimizer("Adam")
def build_adam(params, config):
    return torch.optim.Adam(params, lr=config.get("lr", 1e-3), weight_decay=config.get("weight_decay", 0))

@register_optimizer("SGD")
def build_sgd(params, config):
    return torch.optim.SGD(params, lr=config.get("lr", 0.01), momentum=config.get("momentum", 0.9))

def build_optimizer(model_params, optimizer_config):
    name = optimizer_config["name"]
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer {name} not found in registry.")
    return OPTIMIZER_REGISTRY[name](model_params, optimizer_config)
