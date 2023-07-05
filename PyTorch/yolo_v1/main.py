import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import VOCDataset, train_tfms, inference_tfms
from model import YoloV1, define_optimizer
from loss import YoloLoss
from train import Trainer, StepSchdulerwithWarmup
from params import *


def main(config):
    """
    Training model

    Args:
        config (Config): Training parameters
    """
    if config.use_all:
        print("Use every VOC2012 images")
    else:
        print("Use only one fold of entire VOC2012 images")

    print(f"Training on {config.fold} / {config.nb_splits} fold")

    train_dataset = VOCDataset(img_path=IMAGE_PATH,
                               label_csv=LABEL_CSV,
                               train=True,
                               use_all=config.use_all,
                               fold=config.fold,
                               transform=train_tfms())
    valid_dataset = VOCDataset(IMAGE_PATH,
                               LABEL_CSV,
                               train=False,
                               use_all=config.use_all,
                               fold=config.fold,
                               transform=None)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_workers,
                              pin_memory=True)
    
    model = YoloV1().to(config.device)
    criterion = YoloLoss()

    kwargs = {
        'lr': config.lr,
        'weight_decay': config.decay
    }
    if config.optimizer == 'SGD':
        kwargs['momentum'] = config.momentum

    optimizer = define_optimizer(config.optimizer,
                                 model.parameters(),
                                 **kwargs)
    scheduler = StepSchdulerwithWarmup(optimizer,
                                       total_step=len(train_loader)*135,
                                       warmup_rate=config.warmup_rate,
                                       dataloader=train_loader,
                                       total_epochs=config.epochs)

    trainer = Trainer(config,
                      train_loader,
                      valid_loader,
                      model,
                      criterion,
                      optimizer,
                      scheduler)
    
    trainer.fit()