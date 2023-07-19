import time

import torch
from tqdm import tqdm
from IPython.display import clear_output

from utils import intersection_over_union, save_log


class Trainer:
    def __init__(self,
                 config,
                 train_loader,
                 valid_loader,
                 model,
                 loss_fn,
                 optimizer):
        
        self.epochs = config.epochs
        self.early_stop = config.early_stop
        self.patience = config.early_stop_patience
        self.log_dir = config.log_dir
        self.device = config.device
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
    
    def iteration(self):
        train_loss = 0
        train_iou = 0

        valid_loss = 0
        valid_iou = 0

        self.model.train()
        for imgs, msks in self.train_loader:
            imgs, msks = imgs.to(self.device), msks.to(self.device)
            
            outs = self.model(imgs)
            loss = self.loss_fn(outs, msks)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_iou += intersection_over_union(outs, msks)

        self.model.eval()
        with torch.inference_mode():
            for imgs, msks in self.valid_loader:
                imgs, msks = imgs.to(self.device), msks.to(self.device)
                
                outs = self.model(imgs)
                loss = self.loss_fn(outs, msks)
                
                valid_loss += loss.item()
                valid_iou += intersection_over_union(outs, msks)

        train_loss /= len(self.train_loader)
        train_iou /= len(self.train_loader)
        valid_loss /= len(self.valid_loader)
        valid_iou /= len(self.valid_loader)

        return train_loss, train_iou, valid_loss, valid_iou

    def fit(self):
        start_time = time.time()
        best_loss = 10**9
        best_loss_iou = 0

        loss_patience = 0

        save_dir = self.log_dir + '/model.pth'

        tk0 = tqdm(range(self.epochs), total=self.epochs)
        for epoch in tk0:
            train_loss, train_iou, valid_loss, valid_iou = self.iteration()
            save_log(self.log_dir, epoch, 'train', train_loss, train_iou)
            save_log(self.log_dir, epoch, 'valid', valid_loss, valid_iou)

            tk0.set_postfix(train_loss=f"{train_loss:.4g}",
                            train_IoU=f"{train_iou:.4g}",
                            valid_loss=f"{valid_loss:.4g}",
                            valid_IoU=f"{valid_iou:.4g}",)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_loss_iou = valid_iou
                loss_patience = 0
                torch.save(self.model.state_dict(), save_dir)
            else:
                loss_patience += 1

            if self.early_stop and (loss_patience > self.patience):
                clear_output()
                print(f"Model doesn't improved for "
                    f"loss: {loss_patience} epochs. "
                    f"Stop training. Total time: {round(time.time() - start_time)}s")
                print(f"Trained model save at {save_dir}")
                break
        
        else:
            clear_output()
            print("Done!")
            print(f"Valid Loss: {best_loss:.4g} | "
                f"Valid IoU: {best_loss_iou:.4g} | Total time: {round(time.time() - start_time)}s")
            print(f"Trained model save at {save_dir}")