import copy

BASE_CONFIG = {
    "dataset": {"name": "SALICON"},
    "loss": {"name": "MSE"},
    "metrics": {"PCC": {}, "JSS": {}, "MSE": {}, "NSS": {}, "AUC_Judd": {}},
    "optimizer": {
        "name": "Adam",
        "lr": 1e-4,
        "weight_decay": 1e-5
    },
    "data_root": "./data/SALICON/",
    "mit_data_root": "./data/MIT1003/",
    "batch_size": 32,
    "num_epochs": 10,
    "num_workers": 16,
    "early_stopping": {"enabled": True, "patience": 2, "min_delta": 0.0, "monitor": "val_loss"},
}

EXPERIMENTS = {
    "baseline": {
        "model": {"name": "BaselineCNN"},
    },
    "multiscale": {
        "model": {"name": "MultiScaleCNN"},
    },
    "multiscale_skip": {
        "model": {"name": "MultiScaleSkipCNN"},
    },
    "transformer": {
        "model": {"name": "TransformerSaliency"},
    },
    "baseline_combined": {
        "model": {"name": "BaselineCNN"},
        "loss": {"name": "combined", "alpha": 0.5},
        "num_epochs": 20,
    },
    "multiscale_combined": {
        "model": {"name": "MultiScaleCNN"},
        "loss": {"name": "combined", "alpha": 0.5},
        "num_epochs": 20,
    },
    "multiscale_skip_combined": {
        "model": {"name": "MultiScaleSkipCNN"},
        "loss": {"name": "combined", "alpha": 0.5},
        "num_epochs": 20,
    },
    "transformer_combined": {
        "model": {"name": "TransformerSaliency"},
        "loss": {"name": "combined", "alpha": 0.5},
        "num_epochs": 20,
    },
}

import collections.abc

def deep_update(d, u):
    """Recursively update a nested dictionary."""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# Merge base config into all experiments
for exp_name, exp_config in EXPERIMENTS.items():
    merged_config = copy.deepcopy(BASE_CONFIG)
    deep_update(merged_config, exp_config)
    EXPERIMENTS[exp_name] = merged_config



class ConfigNode:
    def __init__(self, d):
        self.__dict__.update({k: ConfigNode(v) if isinstance(v, dict) else v for k, v in d.items()})

    def __getitem__(self, key):
        return self.__dict__[key]

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

def get_config(name):
    if name not in EXPERIMENTS:
        raise ValueError(f"Config '{name}' not found!")
    
    return ConfigNode(EXPERIMENTS[name])
