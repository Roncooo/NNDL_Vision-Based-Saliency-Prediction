EXPERIMENTS = {
    "baseline": {
        "dataset": {"name": "SALICON"},
        "model": {"name": "SimpleUNet"},
        "loss": {"name": "MSE"},
        "metrics": {"PCC": {}, "JSS": {}, "MSE": {}}, # as a dictionary to allow parameters
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "weight_decay": 1e-5
        },
        "batch_size": 32,
        "num_epochs": 10,
        "num_workers": 4
    }
}


# Wrapper class to allow dot notation access (e.g. config.metrics)
class ConfigNode:
    def __init__(self, d):
        self.__dict__.update({k: ConfigNode(v) if isinstance(v, dict) else v for k, v in d.items()})

def get_config(name):
    if name not in EXPERIMENTS:
        raise ValueError(f"Config '{name}' not found!")
    
    return ConfigNode(EXPERIMENTS[name])
