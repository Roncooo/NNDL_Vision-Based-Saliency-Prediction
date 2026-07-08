import os
import torch
import numpy as np
from PIL import Image
from custom_transforms import PairedTransforms
from torch.utils.data import Dataset, DataLoader

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

print("SaliencyDataset class defined.")

def create_dataloaders(cfg):
    """DataLoader SALICON"""
    
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
    
    train_dataset = SaliencyDataset(train_images, train_maps, IMAGE_TRAIN_PATH, MAP_TRAIN_PATH, transform=train_transforms)
    val_dataset = SaliencyDataset(val_images, val_maps, IMAGE_VAL_PATH, MAP_VAL_PATH, transform=val_transforms)    

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

def create_mit_test_loader(cfg):
    """DataLoader test on MIT1003"""
    
    mit_root = cfg["mit_data_root"]
    IMAGE_DIR = os.path.join(mit_root, "ALLSTIMULI")
    MAP_DIR = os.path.join(mit_root, "ALLFIXATIONMAPS")
    
    raw_images = sorted([f for f in os.listdir(IMAGE_DIR) if not f.startswith('.')])
    all_maps = [f for f in os.listdir(MAP_DIR) if not f.startswith('.')]
    
    test_images = []
    test_maps = []
    
    for img_name in raw_images:
        base_name = os.path.splitext(img_name)[0]
        
        matching_maps = [m for m in all_maps if m.startswith(base_name) and "fixMap" in m]
        
        if matching_maps:
            test_images.append(img_name)
            test_maps.append(matching_maps[0])
        else:
            print(f"'fixMap' not found for image{img_name}")

    assert len(test_images) == len(test_maps), "Error: The number of test images and maps do not match."
    assert len(test_images) > 0, "Error: No corresponding maps found! Check the file names."
    
    test_transforms = PairedTransforms(target_size=(256, 256), is_train=False)
    
    mit_dataset = SaliencyDataset(
        image_files=test_images, 
        map_files=test_maps, 
        image_dir=IMAGE_DIR, 
        map_dir=MAP_DIR, 
        transform=test_transforms
    )
    
    mit_loader = DataLoader(
        mit_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        num_workers=cfg["num_workers"], 
        pin_memory=True
    )
    
    return mit_loader

print("\n--- Data Preparation Complete ---")