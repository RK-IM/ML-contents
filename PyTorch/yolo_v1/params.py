from pathlib import Path

CLASS_IDX = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}
IDX_CLASS = {v:k for k, v in CLASS_IDX.items()}

DATA_PATH = Path('../VOCdevkit/VOC2012/')
IMAGE_PATH = DATA_PATH / 'JPEGImages'
ANNOT_PATH = DATA_PATH / 'Annotations'
LABEL_CSV = DATA_PATH / 'bbox_dataframe.csv'
LOG_DIR = './logs/'

NB_SPLITS = 5
VALID_RATE = 0.2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

S = 7
B = 2
C = 20