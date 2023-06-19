import os
import random
import json
from pathlib import Path
import torch

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


###############
##### RLE #####
###############

def rle_decode(enc, shape):
    img = np.zeros(np.prod(shape), dtype=np.uint8)
    enc = enc.split()
    
    start, length = [np.array(x, dtype=int) 
                     for x in (enc[::2], enc[1::2])]
    start -= 1
    end = start + length
    for lo, hi in zip(start, end):
        img[lo : hi] = 1
    
    return img.reshape(shape, order='F')


def rle_encode(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    enc = np.where(pixels[1:] != pixels[:-1])[0] + 1
    enc[1::2] -= enc[::2]
    return ' '.join(str(x) for x in enc)


def rle_encode_less_memory(mask):
    pixels = mask.flatten(order='F')
    pixels[0] = 0
    pixels[-1] = 0
    enc = np.where(pixels[1:] != pixels[:-1])[0] + 2
    enc[1::2] -= enc[::2]
    return ' '.join(str(x) for x in enc)
    

#########################
##### Image process #####
#########################

def make_slices(dataset, window=1024, overlap=128):
    x, y = dataset.shape
    nx = x // (window - overlap) + 1
    ny = y // (window - overlap) + 1

    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=int)
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=int)

    x1[-1] = x - window
    y1[-1] = y - window

    x2 = (x1 + window).clip(0, x)
    y2 = (y1 + window).clip(0, y)

    slices = np.zeros((nx, ny, 4), dtype=int)
    for i in range(nx):
        for j in range(ny):
            slices[i, j] = x1[i], x2[i], y1[j], y2[j]

    return slices.reshape(-1, 4)
    
    
def load_image_from_slice(dataset, slice, window_size=1024):
    x1, x2, y1, y2 = slice
    if dataset.count == 3:
        image = dataset.read([1, 2, 3],
                             window=Window.from_slices((x1, x2),
                                                       (y1, y2)))
        # [channels, H, W] -> [H, W, channels]
        image = np.moveaxis(image, 0, -1) 

    else:
        subdatasets = dataset.subdatasets
        if len(subdatasets) >= 0:
            image = np.zeros((window_size, window_size, len(subdatasets)),
                              dtype=np.uint8)

            for i, subdataset in enumerate(subdatasets):
                with rasterio.open(subdataset) as layer:
                    image[:, :, i] = layer.read(
                        1, window=Window.from_slices((x1, x2),
                                                     (y1, y2)))
    
    return image


def is_null_image(image,
                  s_limit=(40, 220),
                  threshold=0.05):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    true_pixels = np.sum((s > s_limit[0]) & (s < s_limit[1]))
    proportion = true_pixels / np.prod(s.shape)
    
    return proportion < threshold


####################
###### Metric ######
####################

def dice(out, true, threshold=0.5):
    pred = out.sigmoid() > threshold
    pr = pred.sum()
    gt = true.sum()
    intersection = (pred * true).sum()
    return (2 * intersection / (pr + gt)).item()


#############################
###### Reproducibility ######
#############################

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


#################
###### Log ######
#################

def save_log(log_dir, fold, nb_splits, phase, epoch, loss, score):
    log_dir = Path(log_dir + f'/results_fold_{fold}_{nb_splits}.txt')
    if not log_dir.is_file():
        with open(log_dir, 'w') as f:
            f.write('fold,epoch,phase,loss,dice\n')
    else:
        with open(log_dir, 'a') as f:
            f.write(f"{fold},{epoch},{phase},{loss},{score}\n")


def save_config(config, path):
    dic = config.__dict__.copy()
    del dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"]

    with open(path, 'w') as f:
        json.dump(dic, f)


##################
###### Load ######
##################

def load_models(model, weight_list, device='cuda'):
    models = []
    for weight_path in weight_list:
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
    return models
