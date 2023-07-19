import glob

import cv2
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from params import IMG_HEIGHT, IMG_WIDTH

class DSBDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path, 
                 train=False, 
                 seed = 42,
                 height=IMG_HEIGHT, 
                 width=IMG_WIDTH,
                 **kwargs):
        self.path = path
        self.image_paths = path + 'images/'
        self.images = sorted(glob.glob(self.image_paths + '*.png'))
        self.mask_paths = path + 'masks/'
        self.masks = sorted(glob.glob(self.mask_paths + '*.png'))

        X_train, X_valid, Y_train, Y_valid = train_test_split(
            self.images, 
            self.masks,
            random_state=seed,
            **kwargs
        ) 

        if train:
            self.images = X_train
            self.masks = Y_train
        else:
            self.images = X_valid
            self.masks = Y_valid

        self.height = height
        self.width = width


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.height, self.width),
                           interpolation=cv2.INTER_AREA) / 255
        image = image.astype(np.float32)
        image = np.moveaxis(image, -1, 0)

        mask = cv2.imread(mask, 0)
        mask =cv2.resize(mask, (self.height, self.width),
                         interpolation=cv2.INTER_NEAREST) / 255
        mask = mask.astype(np.float32)
        mask = mask[np.newaxis, ...]
        
        return image, mask