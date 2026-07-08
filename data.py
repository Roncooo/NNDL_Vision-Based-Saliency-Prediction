import os
import torch
import numpy as np
from PIL import Image
from custom_transforms import PairedTransforms
from torch.utils.data import Dataset, DataLoader, random_split

class SaliencyDataset(Dataset):
    def __init__(self, image_files, map_files, image_dir, map_dir, target_size=(256, 256), is_train=False, transform=None):
        
        self.image_files = image_files
        self.map_files = map_files
        self.image_dir = image_dir
        self.map_dir = map_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        map_path = os.path.join(self.map_dir, self.map_files[idx])

        try:
            image = Image.open(img_path).convert('RGB')
            gt_map = Image.open(map_path).convert('L')
            
            if self.transform is not None:
                image, gt_map = self.transform(image, gt_map)
                
            return image, gt_map
        
        except Exception as e:
            print(f"Errore caricamento {img_path}: {e}")
            raise e 
            # return torch.zeros(3, 224, 224), torch.zeros(1, 224, 224) # 

print("SaliencyDataset class defined.")

def create_dataloaders(cfg):
    """Crea e restituisce i DataLoader usando i path dal config."""
    
    IMAGE_DIR = os.path.join(cfg["data_root"], "images")
    MAP_DIR = os.path.join(cfg["data_root"], "maps")
    
    IMAGE_TRAIN_PATH = os.path.join(IMAGE_DIR, "train")
    MAP_TRAIN_PATH = os.path.join(MAP_DIR, "train")
    IMAGE_VAL_PATH = os.path.join(IMAGE_DIR, "val")
    MAP_VAL_PATH = os.path.join(MAP_DIR, "val")
    
    train_images = sorted(os.listdir(IMAGE_TRAIN_PATH))
    train_maps = sorted(os.listdir(MAP_TRAIN_PATH))
    val_images_full = sorted(os.listdir(IMAGE_VAL_PATH))
    val_maps_full = sorted(os.listdir(MAP_VAL_PATH))
    
    train_transforms = PairedTransforms(target_size=(256, 256), is_train=True)
    val_transforms = PairedTransforms(target_size=(256, 256), is_train=False)
    
    train_dataset = SaliencyDataset(train_images, train_maps, IMAGE_TRAIN_PATH, MAP_TRAIN_PATH, transform=train_transforms)
    full_val_dataset = SaliencyDataset(val_images_full, val_maps_full, IMAGE_VAL_PATH, MAP_VAL_PATH, transform=val_transforms)    
    
    total_val_size = len(full_val_dataset)
    test_size = total_val_size // 2
    val_size = total_val_size - test_size

    generator = torch.Generator().manual_seed(42)
    val_dataset, test_dataset = random_split(full_val_dataset, [val_size, test_size], generator=generator)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=True, 
        num_workers=cfg["num_workers"], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        num_workers=cfg["num_workers"], 
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        num_workers=cfg["num_workers"], 
        pin_memory=True)
    
    return train_loader, val_loader, test_loader
print("\n--- Data Preparation Complete ---")