import cv2
import pandas as pd

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from params import MEAN, STD, IMAGE_CSV


class HuBDataset(Dataset):
    def __init__(self, train=True, fold=0, transform=None):
        super().__init__()
        self.csv = pd.read_csv(IMAGE_CSV)
        self.train = train
        if self.train:
            self.df = self.csv[self.csv['fold'] != fold]
            self.df = self.df.reset_index(drop=True)
        else:
            self.df = self.csv[self.csv['fold'] == fold]
            self.df = self.df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        image_path = self.df.loc[idx, 'file_path']
        mask_path = image_path.replace('images', 'masks')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0) / 255.

        if self.transform is not None:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
        else:
            tfms = ToTensorV2()
            totensor = tfms(image=image, mask=mask)
            image = totensor['image']
            mask = totensor['mask']
        mask = mask.unsqueeze(0).float()
        
        return image, mask


def train_tfms(window=1024, reduce=4):
    return A.Compose(
        [
            A.Resize(window//reduce, window//reduce),

            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625,
                            scale_limit=0.2,
                            rotate_limit=15,
                            p=0.5,
                            border_mode=cv2.BORDER_REFLECT),
            
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(hue_shift_limit=30,
                                sat_shift_limit=30,
                                val_shift_limit=30,
                                p=0.5),
            A.CLAHE(p=0.5),

            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.3),

            # A.CoarseDropout(max_holes=8,
            #                 max_height=16,
            #                 max_width=16,
            #                 fill_value=0,
            #                 mask_fill_value=0,
            #                 p=0.2),

            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]
    )


def val_tfms(window=1024, reduce=4):
    return A.Compose(
        [
            A.Resize(window//reduce, window//reduce),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ]
    )