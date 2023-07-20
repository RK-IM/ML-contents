import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DSBDataset
from model import Unet
from train import Trainer


def main(config):
    train_ds = DSBDataset(path=config.data_path,
                          train=True,
                          seed=config.seed,
                          height=config.height,
                          width=config.width)
    
    valid_ds = DSBDataset(path=config.data_path,
                          train=False,
                          seed=config.seed,
                          height=config.height,
                          width=config.width)
    
    train_loader = DataLoader(train_ds,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_ds,
                              batch_size=config.batch_size,
                              shuffle=False,
                              num_workers=config.num_workers,
                              pin_memory=True)
    
    model = Unet(3, nb_classes=1).to(config.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr)
    
    trainer = Trainer(config,
                      train_loader,
                      valid_loader,
                      model,
                      criterion,
                      optimizer)
    trainer.fit()

    
