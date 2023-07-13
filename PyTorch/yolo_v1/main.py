from torch.utils.data import DataLoader

from dataset import VOCDataset, train_tfms
from model import YoloV1
from loss import YoloLoss
from train import Trainer
from optim import define_optimizer, StepLRwithWarmup, ExponentialLRwithWarmup
from transformers import get_cosine_schedule_with_warmup
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
    # scheduler = StepLRwithWarmup(optimizer,
    #                              epochs=config.epochs,
    #                              iter_per_epoch=len(train_loader),
    #                              warmup_rate=config.warmup_rate)
    scheduler = ExponentialLRwithWarmup(optimizer,
                                        epochs=config.epochs,
                                        iter_per_epoch=len(train_loader),
                                        warmup_rate=config.warmup_rate)

    # training_steps = len(train_loader) * config.epochs
    # warmup_steps = int(config.warmup_rate * training_steps)
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=training_steps
    #     )
    
    trainer = Trainer(config,
                      train_loader,
                      valid_loader,
                      model,
                      criterion,
                      optimizer,
                      scheduler)
    
    trainer.fit()