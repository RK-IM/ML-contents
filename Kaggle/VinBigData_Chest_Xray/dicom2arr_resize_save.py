import os
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


DATA_PATH = "../data"
TRAIN_PATH = "test"
train_dicoms = glob.glob(os.path.join(DATA_PATH, TRAIN_PATH) + "/*.dicom")


def dicom2array(path):
    ds = pydicom.read_file(path)
    data = ds.pixel_array
    data = apply_voi_lut(data, ds).astype(np.float32)

    # win_min = ds.WindowCenter - ds.WindowWidth / 2
    # win_max = ds.WindowCenter + ds.WindowWidth / 2
    # data = data.clip(win_min) - win_min
    data /= 2 ** ds.BitsStored

    if ds.PhotometricInterpretation == "MONOCHROME1":
        data = 1 - data

    return data


def resize_save(f_list, size=1024):
    original_dict = {
        "image_id": [],
        "width": [],
        "height": [],
    }
    save_path = f"{DATA_PATH}/{TRAIN_PATH}_{size}"
    print(f"File save at {save_path}")
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for file in tqdm(f_list, total=len(f_list)):
        arr = dicom2array(file)
        im_id = Path(file).stem
        h, w = arr.shape
        original_dict["image_id"].append(im_id)
        original_dict["height"].append(h)
        original_dict["width"].append(w)
        arr_resized = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        arr_resized = (arr_resized * 255).astype(np.uint8)
        cv2.imwrite(save_path + "/" + im_id + ".png", arr_resized)

    return original_dict


if __name__ == "__main__":
    original_dict = resize_save(train_dicoms, size=512)
    # resize_save(train_dicoms, size=512)

    original_df = pd.DataFrame.from_dict(original_dict)
    original_df.to_csv(f"{DATA_PATH}/{TRAIN_PATH}_meta.csv", index=False)
