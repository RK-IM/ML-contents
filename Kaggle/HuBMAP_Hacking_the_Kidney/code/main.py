import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from dataset import HuBDataset, train_tfms, val_tfms
from models import define_model
from train import Trainer
from utils import seed_worker


def main(config):
    print(f":::  Fold {config.fold} / {config.nb_splits}  :::")

    train_ds = HuBDataset(train=True, fold=config.fold, 
                          transform=train_tfms(window=config.window, reduce=config.reduce))
    valid_ds = HuBDataset(train=False, fold=config.fold, 
                          transform=val_tfms(window=config.window))

    # g = torch.Generator()
    # g.manual_seed(0)

    train_loader = DataLoader(train_ds,
                              batch_size=config.train_batch_size,
                              shuffle=True,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                            #   generator=g
                              )
    valid_loader = DataLoader(valid_ds,
                              batch_size=config.valid_batch_size,
                              shuffle=False,
                              num_workers=config.num_workers,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                            #   generator=g
                              )
    
    model = define_model(decoder_name=config.decoder_name,
                         encoder_name=config.encoder_name,
                         num_classes=1,
                         encoder_weights=config.encoder_weights).to(config.device)
    
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.lr)
    
    num_warmup_steps = int(config.warmup_prop * config.epochs * len(train_loader))
    num_training_steps = int(config.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        model=model,
        loss_fn=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer.fit()