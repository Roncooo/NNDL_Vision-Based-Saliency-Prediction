import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # For data augmentation and tensor conversion
import torchvision.transforms.functional as TF # For applying same random transform
from PIL import Image
import os
import numpy as np
import random # For random augmentations
from custom_transforms import PairedTransforms
import matplotlib.pyplot as plt
from config import get_config


class SaliencyDataset(Dataset):
    def __init__(self, image_files, map_files, image_dir, map_dir, target_size=(224, 224), is_train=False, transform=None):
        
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

        # Apri i file dal disco grezzi
        try:
            image = Image.open(img_path).convert('RGB')
            gt_map = Image.open(map_path).convert('L')
            
            # Delega TUTTO il lavoro di modifica all'oggetto esterno
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
    val_images = sorted(os.listdir(IMAGE_VAL_PATH))
    val_maps = sorted(os.listdir(MAP_VAL_PATH))
    
    train_transforms = PairedTransforms(target_size=(256, 256), is_train=True)
    val_transforms = PairedTransforms(target_size=(256, 256), is_train=False)
    
    # Crea i Dataset
    train_dataset = SaliencyDataset(train_images, train_maps, IMAGE_TRAIN_PATH, MAP_TRAIN_PATH, transform=train_transforms)
    val_dataset = SaliencyDataset(val_images, val_maps, IMAGE_VAL_PATH, MAP_VAL_PATH, transform=val_transforms)
    
    # Crea i DataLoader
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
    
    return train_loader, val_loader

# debug
'''
print("\n--- Data Preparation Complete ---")

if __name__ == "__main__":
    print("\n--- TEST FINALE DATALOADER ---")
    
    # Invece di chiamare ConfigNode() vuoto, chiediamo la configurazione "baseline"
    cfg = get_config("baseline") 
    
    try:
        train_loader, val_loader = create_dataloaders(cfg)
        
        # Estrai un BATCH intero
        images, maps = next(iter(train_loader))
        
        print(f"Batch Immagini: {images.shape} | Tipo: {images.dtype}")
        print(f"Batch Mappe:    {maps.shape} | Tipo: {maps.dtype}")
        print("TUTTO FUNZIONA PERFETTAMENTE! ")
        
    except Exception as e:
        print(f"Errore durante il test del DataLoader: {e}")
'''