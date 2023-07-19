import json
from pathlib import Path

import numpy as np


def rle_decode(enc, shape):
    """
    Decode RLE to 2D array mask. As written in competition overview page,
    the pixels are numbered from top to bottom, reshape array using
    column-wise order.

    Args:
        enc (str): Run length encoding.
        shape (tuple[H, W]): Target image size to decode RLE.

    Returns:
        (numpy.array[H, W]): 2D array mask.
    """
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
    """
    Encode binary masked image to run-length encoding.

    Args:
        mask(numpy.array[H, W]): 2D array mask

    Returns:
        (str): RLE
    """
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    enc = np.where(pixels[1:] != pixels[:-1])[0] + 1
    enc[1::2] -= enc[::2]
    return ' '.join(str(x) for x in enc)


def intersection_over_union(pred, true, threshold=0.5):
    """
    Calculate IoU pixel wise

    Args:
        pred (torch.Tensor)
        true (torch.Tensor)
        threshold (float, optional)

    Return:
        (float)
    """
    pred = pred.sigmoid() > threshold
    intersection = pred * true
    union = (pred + true).clip(0, 1)
    
    return (intersection.sum() / union.sum()).cpu().item()


def save_log(log_dir, epoch, phase, loss, score):
    log_dir = Path(log_dir + f"/results.txt")
    if not log_dir.is_file():
        with open(log_dir, 'w') as f:
            f.write('epoch,phase,loss,IoU\n')
            f.write(f'{epoch},{phase},{loss},{score}\n')
    else:
        with open(log_dir, 'a') as f:
            f.write(f'{epoch},{phase},{loss},{score}\n')


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