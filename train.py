import json
import os
import re
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import create_dataloaders
from models import build_model
from metrics import build_metrics
from config import get_config
from losses import build_loss
from optimizers import build_optimizer


def find_latest_checkpoint(experiment, ckpt_dir="checkpoints"):
    """Newest checkpoint belonging to EXACTLY this experiment.

    Do not replace this with glob("best_model_{experiment}_*.pth"): for
    experiment="multiscale" that pattern ALSO matches
    best_model_multiscale_skip_*.pth (the * swallows "skip_"), so it can
    hand back a different model's weights - and since multiscale_skip is the
    newer file, sorting by mtime picks it. Here the timestamp suffix is
    matched explicitly so only true siblings can match.
    """
    pattern = re.compile(rf"^best_model_{re.escape(experiment)}_\d{{8}}_\d{{6}}\.pth$")
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if pattern.match(f)]
    if not files:
        raise FileNotFoundError(f"No checkpoint for '{experiment}' in {ckpt_dir}/")
    return max(files, key=os.path.getmtime)


def save_checkpoint(path, model, optimizer, epoch, best_val_loss, config_name):
    """Save a resumable checkpoint.

    Stores the optimizer state too, so a resumed run continues with Adam's
    moment estimates intact instead of resetting them (which causes a
    transient disruption). Older checkpoints are bare state_dicts and lack
    this - load_checkpoint() handles both formats.
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,                    # epochs completed so far (cumulative)
        "best_val_loss": best_val_loss,
        "config_name": config_name,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    """Load a checkpoint into `model` (and `optimizer` if given).

    Backward compatible on purpose: checkpoints written before this change
    are bare state_dicts (an OrderedDict of tensors) with no optimizer /
    epoch / best_val_loss. Both formats load; the caller gets back whatever
    metadata was available.

    Returns (epoch, best_val_loss) - (0, inf) for old-format checkpoints.
    """
    ck = torch.load(path, map_location=device, weights_only=False)

    # new wrapper format
    if isinstance(ck, dict) and "model_state_dict" in ck:
        model.load_state_dict(ck["model_state_dict"])
        if optimizer is not None and ck.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        return ck.get("epoch", 0), ck.get("best_val_loss", float("inf"))

    # legacy format: bare state_dict
    model.load_state_dict(ck)
    if optimizer is not None:
        print(
            "WARNING: legacy checkpoint (no optimizer state) - Adam moment "
            "estimates start from zero. Expect a small transient bump in the "
            "first epochs after resuming."
        )
    return 0, float("inf")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Handles one epoch of training."""
    model.train()
    total_loss = 0.0
    
    # tqdm for the progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for images, gts in progress_bar:

        # move data to device
        images, gts = images.to(device), gts.to(device)
        
        # forward pass
        preds = model(images) 
        
        # compute loss
        loss = criterion(preds, gts)
        
        # backward pass & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, criterion, metrics_dict, device):
    """Handles evaluation on the validation set."""
    model.eval()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc="Validating")
    results_metrics = {}
    
    for images, gts in progress_bar:
        images, gts = images.to(device), gts.to(device)
        
        preds = model(images)
        loss = criterion(preds, gts)
        total_loss += loss.item()
        
        # iterate over registered metrics
        for metric_name, metric_fn in metrics_dict.items():
            if metric_name not in results_metrics:
                results_metrics[metric_name] = 0.0
            results_metrics[metric_name] += metric_fn(preds, gts)
            
    # compute final metrics over the whole epoch
    results = {"val_loss": total_loss / len(dataloader)}
    for name, val in results_metrics.items():
        results[name] = val / len(dataloader)
    
    return results

def main(config_name, data_root=None, mit_data_root=None, resume_from=None,
         completed_epochs=None, epochs=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # load configuration
    config = get_config(config_name)

    # Override paths if provided
    if data_root is not None:
        config.data_root = data_root
    if mit_data_root is not None:
        config.mit_data_root = mit_data_root
    # epoch budget override (keeps config.py canonical for from-scratch runs;
    # for a continuation this is the MAX number of ADDITIONAL epochs)
    if epochs is not None:
        config.num_epochs = epochs

    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # build everything from the config
    train_loader, val_loader = create_dataloaders(config)

    model = build_model(config.model).to(device)
    criterion = build_loss(config.loss).to(device)
    metrics = build_metrics(config.metrics)
    optimizer = build_optimizer(model.parameters(), config.optimizer)

    best_val_loss = float('inf')
    start_epoch = 0
    train_losses = []
    val_losses = []
    os.makedirs('checkpoints', exist_ok=True)

    # optionally continue a previous run
    if resume_from is not None:
        start_epoch, best_val_loss = load_checkpoint(resume_from, model, optimizer, device)
        if completed_epochs is not None:      # for legacy ckpts that don't record it
            start_epoch = completed_epochs
        print(f"Resumed from {resume_from} (epochs completed: {start_epoch})")

        if best_val_loss == float('inf'):
            # A legacy checkpoint doesn't store best_val_loss. Leaving it at inf
            # would make the FIRST epoch count as "best" even if it were worse
            # than the checkpoint we resumed from - so measure the resumed model
            # and anchor early stopping / best-checkpointing to its real score.
            print("Legacy checkpoint: measuring its val loss to anchor the baseline...")
            baseline = validate(model, val_loader, criterion, metrics, device)
            best_val_loss = baseline['val_loss']
            print(f"Baseline val loss of resumed model: {best_val_loss:.4f} "
                  f"(a new checkpoint is only saved if it beats this)")

    # early stopping settings (num_epochs is a MAX budget once this is enabled)
    es_config = config.get("early_stopping", {})
    es_enabled = es_config.get("enabled", False) if es_config else False
    es_patience = es_config.get("patience", 3) if es_config else 3
    es_min_delta = es_config.get("min_delta", 0.0) if es_config else 0.0
    epochs_no_improve = 0
    best_state = None
    if es_enabled:
        print(f"Early stopping: monitor=val_loss, patience={es_patience}, min_delta={es_min_delta}")

    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{start_epoch + config.num_epochs} ---")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_results = validate(model, val_loader, criterion, metrics, device)

        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_results.items() if k != 'val_loss'])
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_results['val_loss']:.4f} | {metrics_str}")

        train_losses.append(train_loss)
        val_losses.append(val_results)

        # save checkpoints if val_loss improves
        if val_results['val_loss'] < best_val_loss - es_min_delta:
            best_val_loss = val_results['val_loss']
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint_path = os.path.join('checkpoints', f'best_model_{config_name}_{timestamp}.pth')
            save_checkpoint(checkpoint_path, model, optimizer, epoch + 1, best_val_loss, config_name)
            print(f"Saved new best model to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if es_enabled:
                print(f"No improvement for {epochs_no_improve}/{es_patience} epoch(s) "
                      f"(best val loss: {best_val_loss:.4f})")
                if epochs_no_improve >= es_patience:
                    print(f"Early stopping triggered after epoch {epoch+1}: "
                          f"val_loss did not improve for {es_patience} consecutive epochs.")
                    break

    # return the BEST model, not the last one, so it matches the saved checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best weights (val loss: {best_val_loss:.4f})")

    history = {"train_losses": train_losses, "val_losses": val_losses}

    # Persist the history. run.ipynb keeps it in memory, but running train.py
    # from the terminal used to throw it away - so the curves could only be
    # recovered by re-parsing stdout. Saving it here means every run is
    # plottable afterwards, however it was launched.
    os.makedirs("history", exist_ok=True)
    history_path = os.path.join("history", f"{config_name}_{timestamp}.json")
    with open(history_path, "w") as f:
        json.dump({
            **history,
            "config_name": config_name,
            "start_epoch": start_epoch,          # 0 for a fresh run
            "resumed_from": resume_from,
            "best_val_loss": best_val_loss,
            "early_stopped": epochs_no_improve >= es_patience if es_enabled else False,
        }, f, indent=2)
    print(f"Saved training history to {history_path}")

    return model, history

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Saliency Prediction Model")
    parser.add_argument("--config", type=str, default="baseline", help="Experiment config name")
    parser.add_argument("--data-root", type=str, default=None, help="Path to SALICON dataset")
    parser.add_argument("--mit-data-root", type=str, default=None, help="Path to MIT1003 dataset")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint to continue training from")
    parser.add_argument("--completed-epochs", type=int, default=None,
                        help="Epochs already done (only needed for legacy checkpoints, "
                             "which don't record it: e.g. 10 for multiscale, 8 for baseline)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config num_epochs. When resuming, this is the MAX "
                             "number of ADDITIONAL epochs to run.")

    args = parser.parse_args()
    main(args.config, args.data_root, args.mit_data_root, args.resume_from,
         args.completed_epochs, args.epochs)
