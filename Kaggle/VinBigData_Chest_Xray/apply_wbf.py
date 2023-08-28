import os

import numpy as np
import pandas as pd

from tqdm import tqdm

from ensemble_boxes import weighted_boxes_fusion


DATA_DIR = "../data"
TRAIN_IMG_PATH = "train_1024"
TRAIN_META = "train_meta.csv"
TRAIN_DF = "train.csv"

TRAIN_IMGS = os.listdir(os.path.join(DATA_DIR, TRAIN_IMG_PATH))


def main():
    train_meta = pd.read_csv(os.path.join(DATA_DIR, TRAIN_META))
    train_df = pd.read_csv(os.path.join(DATA_DIR, TRAIN_DF))

    class_idx_name = (
        train_df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id")
    )
    class_idx_name = dict(zip(class_idx_name["class_id"], class_idx_name["class_name"]))

    abnormal = train_df[train_df["class_id"] != 14]
    abnormal_ids = abnormal["image_id"].unique()

    wbf = []
    for abnormal_id in tqdm(abnormal_ids):
        ab_df = train_df[train_df["image_id"] == abnormal_id]
        original_size = (
            train_meta.query("image_id==@abnormal_id").values[0][1:].astype(int)
        )  # width, height
        bboxes = ab_df[["x_min", "y_min", "x_max", "y_max"]].values
        bboxes_norm = bboxes / np.stack([original_size] * 2).reshape(-1)
        labels = ab_df["class_id"].values

        bboxes_wbf, _, labels_wbf = weighted_boxes_fusion(
            [bboxes_norm], [np.ones(len(labels))], [labels], iou_thr=0.4
        )
        bboxes_wbf *= np.stack([original_size] * 2).reshape(-1)
        bboxes_wbf = bboxes_wbf.astype(int)

        for i in range(len(bboxes_wbf)):
            wbf.append(
                [
                    abnormal_id,
                    class_idx_name[labels_wbf[i]],
                    labels_wbf[i].astype(int),
                    bboxes_wbf[i][0],
                    bboxes_wbf[i][1],
                    bboxes_wbf[i][2],
                    bboxes_wbf[i][3],
                ]
            )

    wbf_df = pd.DataFrame(
        wbf,
        columns=[
            "image_id",
            "class_name",
            "class_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ],
    )

    normal_df = (
        train_df[~train_df["image_id"].isin(abnormal_ids)]
        .drop_duplicates("image_id")
        .reset_index(drop=True)
        .drop("rad_id", axis=1)
    )

    preprocessed_df = (
        pd.concat([normal_df, wbf_df], axis=0)
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    preprocessed_df.to_csv(os.path.join(DATA_DIR, "train_wbf.csv"), index=False)


if __name__ == "__main__":
    main()
