import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


TRAIN_PATH = './data/stage1_train/'
SAVE_STAGE1_TRAIN_PATH = './data/stage1_train_merged/'

train_ids = next(os.walk(TRAIN_PATH))[1]


def make_dirs(parent):
    parent = Path(parent)
    image_path = parent / 'images'
    mask_path = parent / 'masks'

    if not image_path.is_dir():
        image_path.mkdir(exist_ok=True, parents=True)
    if not mask_path.is_dir():
        mask_path.mkdir(exist_ok=True, parents=True)


def main():
    make_dirs(SAVE_STAGE1_TRAIN_PATH)

    for train_id in tqdm(train_ids, total=len(train_ids)):
        path = TRAIN_PATH + train_id
        image = cv2.imread(path + '/images/' + train_id + '.png')
        height, width = image.shape[:2]

        mask = np.zeros((height, width))
        mask_paths = path + '/masks/'

        for mask_path in Path(mask_paths).glob('*.png'):
            mask_ = cv2.imread(str(mask_path), 0)
            mask = np.maximum(mask, mask_)
        
        cv2.imwrite(SAVE_STAGE1_TRAIN_PATH + 'images/' + train_id + '.png', image)
        cv2.imwrite(SAVE_STAGE1_TRAIN_PATH + 'masks/' + train_id + '.png', mask)


if __name__ == '__main__':
    # Merge stage 1 test data labels
    stage1_sol = pd.read_csv('./data/stage1_solution.csv')
    stage1_sol['EncodedPixels'] += ' '
    stage1_sol_merged = stage1_sol.groupby('ImageId').agg(
        {'EncodedPixels': 'sum', 'Height': 'max', 'Width': 'max'}
    )
    stage1_sol_merged.to_csv('./data/stage1_solution_merged.csv')

    # Merge stage 1 training data labels
    main()