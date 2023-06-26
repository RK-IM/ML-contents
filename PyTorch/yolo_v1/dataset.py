from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings('ignore')

from params import CLASS_IDX


class VOCDataset(Dataset):
    def __init__(self, img_path, label_csv,
                 size=448,
                 transform=None,
                 S=7, B=2, C=20):
        self.img_list = list(Path(img_path).glob('./*jpg'))
        self.label_csv = pd.read_csv(label_csv)
        self.size = size
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C


    def __len__(self):
        return len(self.img_list)
    

    def __getitem__(self, index):
        img_path = self.img_list[index]
        image, bboxes, class_ids = self._data_gen(img_path)

        if self.transform is not None:
            augmented = self.transform(image=image, 
                                       bboxes=bboxes,
                                       class_ids=class_ids)
            image = augmented['image']
            bboxes = augmented['bboxes']
            class_ids = augmented['class_ids']

        else:
            tfms = A.Compose(
                [
                    A.Resize(self.size, self.size),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids'])
            )
            img_bb_cls = tfms(image=image,
                              bboxes=bboxes,
                              class_ids=class_ids)
            image = img_bb_cls['image']
            bboxes = img_bb_cls['bboxes']
            class_ids = img_bb_cls['class_ids']

        label_matrix = self._label_convert(class_ids, bboxes)
        return image, label_matrix

    
    def _data_gen(self, img_path):
        img_id = img_path.stem

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        labels = self.label_csv.query("id==@img_id")
        labels['class'] = labels['class'].map(CLASS_IDX)

        class_ids = labels['class'].values

        bboxes = labels[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy(dtype=np.float32)
        return image, bboxes, class_ids
    

    def _label_convert(self, class_ids, bboxes):
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for class_id, bbox in zip(class_ids, bboxes):
            xmin, ymin, xmax, ymax = map(lambda x: x/self.size, bbox)

            x_center = (xmin + xmax) / 2
            x_grid = int(self.S * x_center)
            x_pos = self.S * x_center - x_grid

            y_center = (ymin + ymax) / 2
            y_grid = int(self.S * y_center)
            y_pos = self.S * y_center - y_grid

            w = (xmax - xmin) * self.S
            h = (ymax - ymin) * self.S

            label_matrix[x_grid, y_grid, 20] = 1
            box_coordinates = torch.tensor(
                [x_pos, y_pos, w, h]
            )
            label_matrix[x_grid, y_grid, 21:25] = box_coordinates
            label_matrix[x_grid, y_grid, class_id] = 1
        
        return label_matrix



def train_tfms(size=448):
    return A.Compose(
        [
            A.Resize(size, size),
            A.ShiftScaleRotate(shift_limit=0.2,
                               scale_limit=0.2,
                               rotate_limit=0),
            A.ColorJitter(saturation=0.5,
                          hue=0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_ids'])
    )


def inference_tfms(size=448):
    return A.Compose(
        [
            A.Resize(size, size),
            ToTensorV2(),
        ]
    )


if __name__ == '__main__':
    from params import *
    from torch.utils.data import DataLoader

    dataset = VOCDataset(img_path=IMAGE_PATH,
                         label_csv=LABEL_CSV,
                         transform=train_tfms())
    
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)
    
    sample = next(iter(dataloader))
    print(sample[0].shape, sample[1].shape)
