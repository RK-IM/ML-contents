import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from dataset import (
    get_train_transforms,
    get_valid_transforms,
    get_test_transforms,
    EfficientDetDataset,
    EfficientDetDatasetInference,
)


class EfficientDetDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset_adaptor,
            validation_dataset_adaptor,
            predict_dataset_adaptor,
            train_transforms=get_train_transforms(target_img_size=512),
            valid_transforms=get_valid_transforms(target_img_size=512),
            predict_transforms=get_test_transforms(target_img_size=512),
            num_workers=4,
            batch_size=8,
    ):
        super().__init__()
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.predict_ds = predict_dataset_adaptor

        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.predict_tfms = predict_transforms

        self.num_workers = num_workers
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.train_dataset = EfficientDetDataset(
            dataset_adaptor=self.train_ds,
            transforms=self.train_tfms
        )
        self.val_dataset = EfficientDetDataset(
            dataset_adaptor=self.valid_ds, 
            transforms=self.valid_tfms
        )
        self.predict_dataset = EfficientDetDatasetInference(
            dataset_adaptor=self.predict_ds,
            transforms=self.predict_tfms
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return valid_loader
    
    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=self.num_workers,
        )
        return predict_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images).float()
        
        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()
        
        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }
        
        return images, annotations, targets, image_ids