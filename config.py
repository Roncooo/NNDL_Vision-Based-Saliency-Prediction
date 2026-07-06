
EXPERIMENTS = {
    "baseline": {
        "model": {"name": "SimpleUNet"},
        "loss": {"name": "MSE"},
        "metrics": {"PCC": {}, "JSS": {}, "MSE": {}}, # as a dictionary to allow parameters
        "optimizer": {
            "name": "Adam",
            "lr": 1e-4,
            "weight_decay": 1e-5
        },
        "batch_size": 32,
        "num_epochs": 10
    }
}

def get_config(name):
    if name not in EXPERIMENTS:
        raise ValueError(f"Config '{name}' not found!")
    
    return EXPERIMENTS[name]
