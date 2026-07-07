
import random
import torch
import torchvision.transforms.functional as TF

class PairedTransforms:
    #modifiche spaziali (flip, crop, ecc.) devono essere applicate sia all'immagine che alla mappa di salienza
    def __init__(self, target_size=(256, 256), is_train=True):
        self.target_size = target_size
        self.is_train = is_train

    def __call__(self, image, gt_map):
        image = TF.resize(image, self.target_size)
        gt_map = TF.resize(gt_map, self.target_size)

        if self.is_train:
            # Esempio: Random Horizontal Flip (50% di probabilità)
            if random.random() > 0.5:
                image = TF.hflip(image)
                gt_map = TF.hflip(gt_map)
            
            # Aggiungere qui altre trasformazioni

        image = TF.to_tensor(image)
        gt_map = TF.to_tensor(gt_map)

        # Normalizzazione dell'immagine RGB (Standard ImageNet)
        # Nota: La mappa di salienza NON va normalizzata, deve restare tra 0 e 1!
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, gt_map