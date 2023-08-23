import os
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils import show_image


def get_train_transforms(target_img_size=512):
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_test_transforms(target_img_size=512):
    return A.Compose(
        [
            A.Resize(height=target_img_size, width=target_img_size, p=1),
            A.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.,
    )


class CarsDatasetAdaptor:
    def __init__(self, images_dir_path, annotations_dataframe):
        self.images_dir_path = images_dir_path
        self.annotations_df = annotations_dataframe
        self.images = self.annotations_df.image.unique().tolist()
    
    def __len__(self) -> int:
        return len(self.images)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.images_dir_path, image_name))
        pascal_bboxes = self.annotations_df[self.annotations_df.image==image_name][
            ["xmin", "ymin", "xmax", "ymax"]
        ].values
        class_labels = np.ones(len(pascal_bboxes))

        return image, pascal_bboxes, class_labels, index

    def show_image(self, index):
        image, bboxes, class_labels, image_id = self.get_image_and_labels_by_idx(index)
        print(f"image_id: {image_id}")
        show_image(image, bboxes.tolist())
        print(class_labels)


class CarsDatasetAdaptorInference:
    def __init__(self, images_dir_path):
        self.images_dir_path = images_dir_path
        self.images = os.listdir(self.images_dir_path)
    
    def __len__(self):
        return len(self.images)

    def get_image_by_idx(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.images_dir_path, image_name))
        return image, index


class EfficientDetDataset(Dataset):
    def __init__(
        self,
        dataset_adaptor,
        transforms=get_valid_transforms()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self.ds.get_image_and_labels_by_idx(index)

        sample = {
            "image": np.array(image, dtype=np.float32),
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }
        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        labels = sample["labels"]

        _, new_h, new_w = image.shape
        # convert to yxyx
        sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]

        target = {
            "bboxes": torch.as_tensor(sample["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(labels),
            "image_id": torch.tensor([image_id]),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.]),
        }

        return image, target, image_id

    def __len__(self):
        return len(self.ds)


class EfficientDetDatasetInference(Dataset):
    def __init__(
        self,
        dataset_adaptor,
        transforms=get_test_transforms()
    ):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image, image_id = self.ds.get_image_by_idx(index)
        image = np.array(image, dtype=np.float32)
        h, w, _ = image.shape
        image = self.transforms(image=image)["image"]
        return image, (h, w), torch.tensor([image_id])