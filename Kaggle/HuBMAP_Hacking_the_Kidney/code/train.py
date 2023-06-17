import torch
from tqdm.auto import tqdm

from utils import dice


class Trainer():
    def __init__(self,
                 config,
                 train_loader,
                 valid_loader,
                 model,
                 loss_fn,
                 optimizer,
                 scheduler):
        
        self.epochs = config.epochs
        self.early_stop = config.early_stop
        self.patience = config.early_stop_patience
        self.log_dir = config.log_dir
        self.fold = config.fold
        self.device = config.device
        self.use_amp = config.use_amp

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)


    def iteration(self, phase):
        if phase == 'train':
            dataloader = self.train_loader
            self.model.train()
        else:
            dataloader = self.valid_loader
            self.model.eval()

        batches = len(dataloader)
        iter_loss = 0
        iter_dice = 0

        tk0 = tqdm(dataloader, total=batches)
        tk0.set_description(desc=f"{phase}")
        for i, (img, msk) in enumerate(tk0, 1):
            img, msk = img.to(self.device), msk.to(self.device)

            if phase == 'train':
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    out = self.model(img)
                    loss = self.loss_fn(out, msk)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.scheduler.step()
            
            else:
                out = self.model(img)
                loss = self.loss_fn(out, msk)

            iter_loss += loss.item()
            iter_dice += dice(out, msk)

            tk0.set_postfix(lr=f"{self.optimizer.param_groups[0]['lr']:.4g}",
                            loss=f"{iter_loss/i:.4g}",
                            dice=f"{iter_dice/i:.4g}")
        
        return iter_loss / batches, iter_dice / batches
    

    def fit(self):
        best_loss = 10**9
        best_dice = 0

        loss_patience = 0
        dice_patience = 0

        for epoch in range(self.epochs):
            print(f"Epoch: {epoch+1} / {self.epochs}")
            train_loss, train_dice = self.iteration('train')

            with torch.inference_mode():
                valid_loss, valid_dice = self.iteration('valid')

            if valid_loss < best_loss:
                best_loss = valid_loss
                loss_patience = 0
                torch.save(self.model.state_dict(),
                           f"{self.log_dir}/best_loss_fold_{self.fold}.pth")
            else:
                loss_patience += 1

            if valid_dice > best_dice:
                best_dice = valid_dice
                dice_patience = 0
                torch.save(self.model.state_dict(),
                           f"{self.log_dir}/best_dice_fold_{self.fold}.pth")
            else:
                dice_patience += 1

            if self.early_stop and (loss_patience >= self.patience
                                    and dice_patience >= self.patience):
                print(f"Model doesn't improved for {self.patience} epochs. Stop training.")
                break
            print()

        print(f"Done. Best Loss: {best_loss:.4g} | Best Dice: {best_dice:.4g}")