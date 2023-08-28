import os
from PIL import Image

import cv2
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils import show_image


# ------------#
# Transforms #
# ------------#

# Classify
def train_transforms_clf(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size),
            A.RandomScale(scale_limit=0.1),
            A.PadIfNeeded(
                min_height=target_img_size,
                min_width=target_img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            A.RandomCrop(height=target_img_size, width=target_img_size),
            A.RandomBrightnessContrast(p=0.8),
            A.ChannelDropout(p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(p=0.5),
                    A.MedianBlur(blur_limit=5, p=0.5),
                    A.GaussianBlur(p=0.5),
                    A.GaussNoise(p=0.5),
                ],
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


def valid_transforms_clf(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


# Detect
def train_transforms_det(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size),
            A.RandomScale(scale_limit=0.1),
            A.PadIfNeeded(
                min_height=target_img_size,
                min_width=target_img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            A.RandomCrop(height=target_img_size, width=target_img_size),
            A.RandomBrightnessContrast(p=0.8),
            A.ChannelDropout(p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(p=0.5),
                    A.MedianBlur(blur_limit=5, p=0.5),
                    A.GaussianBlur(p=0.5),
                    A.GaussNoise(p=0.5),
                ],
                p=0.5,
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"],),
    )


def valid_transforms_det(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size),
            A.Normalize(),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"],),
    )


# Inference
def pred_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size),
            A.Normalize(),
            ToTensorV2(),
        ]
    )


# ---------#
# Adaptor #
# ---------#
class XrayDatasetAdaptor:
    def __init__(self, images_dir, annotations_df, meta_df):
        self.images_dir = images_dir
        self.annotations_df = annotations_df.copy()
        self.meta_df = meta_df

        self.images = self.annotations_df["image_id"].unique().tolist()

    def __len__(self):
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        return None


# Classify
class XrayClassifyAdaptor(XrayDatasetAdaptor):
    def __init__(self, images_dir, annotations_df, meta_df):
        super().__init__(images_dir, annotations_df, meta_df)

        self.annotations_df["class_id"] = np.where(
            self.annotations_df["class_id"] == 14, 0, 1
        )
        self.annotations_df = self.annotations_df.drop_duplicates(
            "image_id"
        ).reset_index(drop=True)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.images_dir, image_name) + ".png")
        label = self.annotations_df[self.annotations_df["image_id"] == image_name][
            "class_id"
        ].values[0]

        return image, label


# Detect
class XrayDetectAdaptor(XrayDatasetAdaptor):
    def __init__(
        self, images_dir, annotations_df, meta_df, n_splits=5, fold_index=4, train=False
    ):
        super().__init__(images_dir, annotations_df, meta_df)

        self.n_splits = n_splits
        self.fold_index = fold_index

        self.annotations_df = self.annotations_df.query("class_id != 14").reset_index(
            drop=True
        )
        self.add_sgkf_fold(n_splits)

        if train:
            self.annotations_df = self.annotations_df.query("fold!=@fold_index")
        else:
            self.annotations_df = self.annotations_df.query("fold==@fold_index")

        self.images = self.annotations_df["image_id"].unique().tolist()

    def __len__(self):
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.images_dir, image_name) + ".png")
        subset = self.annotations_df.query("image_id==@image_name")
        pascal_bboxes = subset[["x_min", "y_min", "x_max", "y_max"]].values
        class_labels = subset["class_id"].values
        size = self.meta_df.query("image_id==@image_name")[["height", "width"]].values[
            0
        ]
        return image, pascal_bboxes, class_labels, image_name, size

    def show_image(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_name,
            size,
        ) = self.get_image_and_labels_by_idx(index)
        show_image(image, pascal_bboxes, class_labels, image_name, size)

    def add_sgkf_fold(self, n_splits=5):
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        self.annotations_df["fold"] = -1
        for i, (_, fold_idx) in enumerate(
            sgkf.split(
                self.annotations_df,
                y=self.annotations_df["class_id"],
                groups=self.annotations_df["image_id"],
            )
        ):
            self.annotations_df.loc[fold_idx, "fold"] = i


# Inference
class XrayInferenceAdaptor:
    def __init__(self, images_dir, meta_df):
        self.images_dir = images_dir
        self.meta_df = meta_df
        self.images = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.images)

    def get_image_by_index(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.images_dir, image_name))
        image_name = image_name.split(".")[0]
        size = self.meta_df.query("image_id==@image_name")[["height", "width"]].values[
            0
        ]
        return image, image_name, size


# ---------#
# Dataset #
# ---------#

# Classify
class XrayClassifyDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms=valid_transforms_clf()):
        self.adaptor = dataset_adaptor
        self.transforms = transforms

    def __len__(self):
        return len(self.adaptor)

    def __getitem__(self, index):
        image, label = self.adaptor.get_image_and_labels_by_idx(index)
        image = np.stack([image] * 3).transpose(1, 2, 0)  # c, h, w -> h, w, c

        image = self.transforms(image=image)["image"]

        return image, label


# Detect
class XrayDetectDataset(Dataset):
    def __init__(
        self, dataset_adaptor, transforms=valid_transforms_det(), bboxes_yxyx=True
    ):
        super().__init__()
        self.adaptor = dataset_adaptor
        self.transforms = transforms
        self.bboxes_yxyx = bboxes_yxyx

    def __len__(self):
        return len(self.adaptor)

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_name,
            size,
        ) = self.adaptor.get_image_and_labels_by_idx(index)
        image = np.stack([image] * 3).transpose(1, 2, 0)
        # label range 0~13 to 1~14
        class_labels += 1

        sample = {
            "image": image,
            # bbox: xyxy / size: [h, w], original size, image shape: resized
            "bboxes": pascal_bboxes
            / [size[1], size[0], size[1], size[0]]
            * image.shape[0],
            "labels": class_labels,
        }
        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        # xyxy -> yxyx
        if self.bboxes_yxyx:
            sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]

        image = sample["image"]
        target = {
            "bboxes": torch.tensor(sample["bboxes"], dtype=torch.float),
            "labels": torch.tensor(sample["labels"]),
            "img_size": (image.shape[1], image.shape[2]),
            "img_scale": torch.tensor([1.0]),
        }
        return image, target


# Inference
class XrayInferenceDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms=pred_transforms()):
        self.adaptor = dataset_adaptor
        self.transforms = transforms

    def __len__(self):
        return len(self.adaptor)

    def __getitem__(self, index):
        image, image_name, size = self.adaptor.get_image_by_index(index)
        image = np.stack([image] * 3).transpose(1, 2, 0)
        image = self.transforms(image=image)["image"]
        return image, size, image_name
