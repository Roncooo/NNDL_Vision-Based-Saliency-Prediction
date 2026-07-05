import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# TODO
# from data import build_dataset
# from models import build_model
# from metrics import build_metrics
from config import get_config
from losses import build_loss
from optimizers import build_optimizer

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
    
    for images, gts in progress_bar:
        images, gts = images.to(device), gts.to(device)
        
        preds = model(images)
        loss = criterion(preds, gts)
        total_loss += loss.item()
        
        # iterate over registered metrics
        for metric_name, metric_fn in metrics_dict.items():
            # TODO
            pass 
            
    # compute final metrics over the whole epoch
    results = {"val_loss": total_loss / len(dataloader)}
    # results.update({name: fn.compute() for name, fn in metrics_dict.items()})
    
    return results

def main(config_name):
    # load configuration
    config = get_config(config_name)
    
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # build everything from the config
    train_dataset = build_dataset(config.dataset, split='train')
    val_dataset = build_dataset(config.dataset, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    model = build_model(config.model).to(device)
    criterion = build_loss(config.loss).to(device)
    metrics = build_metrics(config.metrics)
    optimizer = build_optimizer(model.parameters(), config.optimizer)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(config.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{config.num_epochs} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_results = validate(model, val_loader, criterion, metrics, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_results['val_loss']:.4f}")
        # TODO other metrics
        
        train_losses.append(train_loss)
        val_losses.append(val_results['val_loss'])

        # save checkpoints here if val_loss improves
        if val_results['val_loss'] < best_val_loss:
            best_val_loss = val_results['val_loss']
            checkpoint_path = os.path.join('checkpoints', f'best_model_{config_name}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")

    return model, {"train_losses": train_losses, "val_losses": val_losses}

if __name__ == "__main__":
    main("baseline")
