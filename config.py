import copy

BASE_CONFIG = {
    "dataset": {"name": "SALICON"},
    "metrics": {"PCC": {}, "JSS": {}, "MSE": {}, "NSS": {}, "AUC_Judd": {}},
    "optimizer": {
        "name": "Adam",
        "lr": 1e-4,
        "weight_decay": 1e-5
    },
    "data_root": "./SALICON/",
    "mit_data_root": "./MIT1003/",
    "batch_size": 32,
    "num_epochs": 10,
    "num_workers": 16
}

EXPERIMENTS = {
    "baseline": {
        "model": {"name": "BaselineCNN"},
        "loss": {"name": "Combined", "alpha": 0.5},
    },
    "multiscale": {
        "model": {"name": "MultiScaleCNN"},
        "loss": {"name": "Combined", "alpha": 0.5},
    },
    "transformer": {
        "model": {"name": "TransformerSaliency"},
        "loss": {"name": "Combined", "alpha": 0.5},
    }
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



# Wrapper class to allow dot notation access (e.g. config.metrics)
# Also supports dict-style access (config["name"], config.get, iteration),
# since losses.py / optimizers.py / metrics.py use that style.
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
