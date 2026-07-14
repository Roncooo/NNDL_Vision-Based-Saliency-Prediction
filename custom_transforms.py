import random
import torchvision.transforms.functional as TF

class PairedTransforms:
    def __init__(self, target_size=(256, 256), is_train=True):
        self.target_size = target_size
        self.is_train = is_train

    def __call__(self, image, gt_map):
        image = TF.resize(image, self.target_size)
        gt_map = TF.resize(gt_map, self.target_size)

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                gt_map = TF.hflip(gt_map)

        image = TF.to_tensor(image)
        gt_map = TF.to_tensor(gt_map)

        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return image, gt_map