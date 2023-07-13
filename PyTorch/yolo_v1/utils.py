import os
import json
import random
from pathlib import Path

import cv2
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import patches

from params import IDX_CLASS, MEAN, STD
cmap = plt.cm.hsv(np.linspace(0, 1, 20))


def seed_everything(seed):
    """
    Set seed value for all random values

    Args:
        seed (int): seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def seed_worker(worker_id):
#     """
#     Set worker seed to preserve reproducibility of generator
#     See 'https://pytorch.org/docs/stable/notes/randomness.html'
#     """
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def save_log(log_dir, use_all, fold, nb_splits, iou, phase, epoch, loss, score):
    """
    Save training logs to log_dir.

    Args:
        log_dir (str): Log directory.
        use_all (bool): Whether use entire dataset or not.
        fold (int): Fold of training data.
        nb_splits (int): Total fold of training dataset.
        iou (float): IOU threshold.
        phase (str): Phase of training. 'train' or 'valid'.
        epoch (int): Epoch of current step.
        loss (float): Loss of current epoch.
        score (float): mAP
    """

    if use_all:
        uses = 'all'
    else:
        uses = 'partial'
    iou = ''.join(str(iou).split('.'))
    log_dir = Path(log_dir 
                   + f"/results_{uses}_fold_{fold}_of_{nb_splits}_iou_{iou}.txt")
    if not log_dir.is_file():
        with open(log_dir, 'w') as f:
            f.write('use_all,fold,epoch,phase,loss,mAP\n')
            f.write(f"{uses},{fold},{epoch},{phase},{loss},{score}\n")
    else:
        with open(log_dir, 'a') as f:
            f.write(f"{uses},{fold},{epoch},{phase},{loss},{score}\n")


def save_config(config, path):
    """
    Save config file to path.

    Args:
        config (Config): Parameters for training.
        path (str): Directory to save config file.
    """
    dic = config.__dict__.copy()
    del dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"]

    with open(path, 'w') as f:
        json.dump(dic, f)


def bbox2rect(bbox, height, width):
    """
    Convert boundary boxes to matplotlib rectangle patch.
    
    Args:
        bbox (list[class_id, confidence, x, y, width, height]):
            Boundary box information
        height (int): Height of original image
        width (int): Width of original image
    
    Return:
        (matplotlib.patches.Rectangle)
    """
    class_id = int(bbox[0])
    x_center = bbox[2] * width
    y_center = bbox[3] * height
    box_width = bbox[4] * width
    box_height = bbox[5] * height

    xmin = x_center - box_width/2
    ymin = y_center - box_height/2

    rect = patches.Rectangle((xmin, ymin),
                             box_width,
                             box_height,
                             facecolor='none',
                             edgecolor=cmap[class_id],
                             linewidth=2)
    return rect


def plot_img_with_bboxes(image, bboxes):
    """
    Plot image with boundary boxes.

    Args:
        image (numpy.array): Original image to plot
        bboxes (list[[class_id, confidence, x, y, width, height]], [], ...):
            Boundary boxes
    """
    plt.imshow(image)
    ax = plt.gca()
    for bbox in bboxes:
        rect = bbox2rect(bbox, image.shape[0], image.shape[1])
        ax.add_patch(rect)
        c = cmap[int(bbox[0])]
        ax.text(rect.get_x(), rect.get_y(), 
                f"{IDX_CLASS[int(bbox[0])]} {bbox[1]:.2f}",
                c='w',
                horizontalalignment='left',
                bbox=(dict(facecolor=cmap[int(bbox[0])],
                        edgecolor='none',
                        pad=1.5)))
    plt.show()