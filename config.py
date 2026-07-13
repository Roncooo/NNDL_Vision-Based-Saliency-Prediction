EXPERIMENTS = {
    "baseline": {
        "dataset": {"name": "SALICON"},
        "model": {"name": "BaselineCNN"},
        "loss": {"name": "combined", "alpha": 0.5},
        "metrics": {"PCC": {}, "JSS": {}, "MSE": {}, "NSS": {}, "AUC_Judd": {}}, # as a dictionary to allow parameters
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
    },
    "multiscale": {
        "dataset": {"name": "SALICON"},
        "model": {"name": "MultiScaleCNN"},
        "loss": {"name": "combined", "alpha": 0.5},
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
}


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
