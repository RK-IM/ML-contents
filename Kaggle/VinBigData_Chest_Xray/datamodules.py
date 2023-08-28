from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from dataset import (
    train_transforms_clf,
    valid_transforms_clf,
    train_transforms_det,
    valid_transforms_det,
    pred_transforms,
    XrayClassifyDataset,
    XrayDetectDataset,
    XrayInferenceDataset,
)


# Classify
class XrayClassifyDataModule(pl.LightningDataModule):
    def __init__(
            self,
            adaptor,
            # pred_adaptor,
            train_transforms=train_transforms_clf(),
            valid_transforms=valid_transforms_clf(),
            # pred_transforms=pred_transforms(),
            n_splits=5,
            fold_index=0,
            batch_size=8,
            num_workers=4,
    ):
        super().__init__()
        self.adaptor = adaptor
        # self.pred_adaptor = pred_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        # self.pred_tfms = pred_transforms

        self.n_splits = n_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
    

    def setup(self, stage=None):
        self.train_dataset = XrayClassifyDataset(
            self.adaptor, self.train_tfms
        )
        self.valid_dataset = XrayClassifyDataset(
            self.adaptor, self.valid_tfms
        )
        # self.pred_dataset = XrayInferenceDataset(
        #     self.pred_adaptor, self.pred_tfms
        # )

        self.train_index, self.valid_index = self.get_skf_index(
            n_splits=self.n_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)

    
    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader
    

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return valid_loader
    
    # def predict_dataloader(self):
    #     pred_loader = DataLoader(
    #         self.pred_dataset,
    #         batch_sampler=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         pin_memory=True
    #     )
    #     return pred_loader


    def get_skf_index(self, n_splits=5, fold_index=0):
        skf = StratifiedKFold(n_splits=n_splits)
        train_fold = []
        valid_fold = []
        for tr_idx, vl_idx in skf.split(self.adaptor.annotations_df,
                                        self.adaptor.annotations_df["class_id"]):
            train_fold.append(tr_idx)
            valid_fold.append(vl_idx)
        
        return train_fold[fold_index], valid_fold[fold_index]


# Detect
class XrayDetectDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_adaptor,
            valid_adaptor,
            train_transforms=train_transforms_det(),
            valid_transforms=valid_transforms_det(),
            n_splits=5,
            fold_index=0,
            num_workers=4,
            batch_size=8,
    ):
        super().__init__()
        self.train_adaptor = train_adaptor
        self.valid_adaptor = valid_adaptor

        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms

        self.n_splits = n_splits
        self.fold_index = fold_index
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = XrayDetectDataset(
            self.train_adaptor, self.train_tfms
        )
        self.valid_dataset = XrayDetectDataset(
            self.valid_adaptor, self.valid_tfms
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return train_loader
    
    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return valid_loader
    
    def collate_fn(self, batch):
        images, targets = tuple(zip(*batch))

        images = torch.stack(images)
        bboxes = [target["bboxes"] for target in targets] # diff length
        labels = [target["labels"] for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets])
        img_scale = torch.tensor([target["img_scale"] for target in targets])

        annotations = {
            "bbox": bboxes, # for effdet box and loss calculation
            "cls": labels, # for effdet box and loss calculation
            "img_size": img_size, # to apply wbf
            "img_scale": img_scale # to apply wbf
        }

        # annotation: effdet model input / dict[list]
        # target: tfms, aggregate and apply WBF / list[dict]
        return images, annotations, targets 


class XrayInferenceDataModule(pl.LightningDataModule):
    def __init__(
            self,
            pred_adaptor,
            pred_transforms=pred_transforms(),
            batch_size=8,
            num_workers=4,
    ):
        super().__init__()
        self.pred_adaptor = pred_adaptor
        self.pred_transforms = pred_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.pred_dataset = XrayInferenceDataset(
            self.pred_adaptor, self.pred_transforms
        )
    
    def predict_dataloader(self):
        pred_loader = DataLoader(
            self.pred_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return pred_loader