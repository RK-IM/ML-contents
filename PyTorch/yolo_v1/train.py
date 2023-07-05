import gc
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from IPython.display import clear_output

from metrics import (
    non_max_suppression,
    mean_average_precision,
    out2boxlist,
)
from utils import save_log


def get_mAP(model, dataloader, device,
            confidence_threshold=0.4,
            nms_threshold=0.5,
            iou_threshold=0.4,
            box_format="midpoint"):
    """
    Calculate mean average precsion using every outputs of model.

    Args:
        model (torch model): Trained model
        dataloader (torch dataloader): Data loaders that contains 
            data to calculate mAP.
        device (torch device): Device to use. 'cuda' if GPU available,
            otherwise use 'cpu'.
        confidence_threshold (float): Use boundary box which has
            higher confidence than this value. Defaults to 0.4.
        nms_threshold (float): Minimal IOU to apply non-maximum
            suppression between predicted boundary box. 
            Defaults to 0.5.
        iou_threshold (float): Select pred_box which has higher iou than
            this value respects to ground truth boundary box.
            Defaults to 0.4.
        box_format (str): 'midpoint' or 'corners', 
            correspond to (x, y, w, h) and (xmin, ymin, xmax, ymax)
    
    Return:
        mAP (float)
    """
    all_true_bboxes = []
    all_pred_bboxes = []
    
    model.eval()
    idx = 0
    with torch.inference_mode():
        for imgs, lbls in dataloader:
            batch_size = imgs.shape[0]
            imgs, lbls = imgs.to(device), lbls.to(device)

            outs = model(imgs)
            true_bboxes = out2boxlist(lbls)
            pred_bboxes = out2boxlist(outs)

            for i in range(batch_size):
                nms_bboxes = non_max_suppression(
                    pred_bboxes[i],
                    confidence_threshold=confidence_threshold,
                    iou_threshold=nms_threshold,
                    box_format=box_format
                )

                for nms_bbox in nms_bboxes:
                    all_pred_bboxes.append([idx]+nms_bbox)
                for bbox in true_bboxes[i]:
                    if bbox[1] > 0.5:
                        all_true_bboxes.append([idx]+bbox)
                idx += 1
    
    mAP = mean_average_precision(all_pred_bboxes,
                                 all_true_bboxes,
                                 iou_threshold=iou_threshold,
                                 box_format=box_format)
    
    return mAP.item()


class StepSchdulerwithWarmup(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear warmup and decayed by 10 at certain step.
    As written in Yolo-v1 paper, learning rate decay when
    epoch is 75 and 105.
    """
    def __init__(self, 
                 optimizer, 
                 total_step, 
                 warmup_rate, 
                 dataloader, 
                 total_epochs=135,
                 last_epoch=-1):
        """
        Constructor
        
        Args:
            optimizer (torch.optim): Optimizer
            total_step (int): total step of training. (epoch * iterations)
            warmup_rate (float): percentage of total steps to 
                warm up learning rate.
            dataloader (torch dataloader): Dataloader to find number of iteration.
            total_epoch (int): Total epochs to train model. Defaults to 135.
            last_epoch (int): Index of last epoch. Defaults to -1
        """
        def lr_lambda(step):
            warmup_step = total_step * warmup_rate
            epoch = total_epochs/135

            if step < warmup_step:
                return float(step + 0.1*warmup_step) / float(max(1., warmup_step)+max(0.1, 0.1*warmup_step))
            else:
                if step < epoch * 75 * len(dataloader):
                    return 1
                elif step < epoch * 105 * len(dataloader):
                    return 0.1
                else:
                    return 1e-2
        
        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class Trainer:
    def __init__(self,
                 config,
                 train_loader,
                 valid_loader,
                 model,
                 loss_fn,
                 optimizer,
                 scheduler):
        
        self.epochs = config.epochs
        self.log_dir = config.log_dir
        self.fold = config.fold
        self.use_all = config.use_all
        if self.use_all:
            self.uses = 'all'
        else:
            self.uses = 'partial'
        self.nb_splits = config.nb_splits
        self.device = config.device
        self.confidence_threshold = config.confidence_threshold
        self.nms_threshold = config.nms_threshold
        self.iou_threshold = config.iou_threshold
        self.early_stop = config.early_stop
        self.loss_patience = config.loss_patience
        self.mAP_patience = config.mAP_patience
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

    
    def iteration(self, phase):
        if phase == 'train':
            dataloader = self.train_loader
            self.model.train()
        else:
            dataloader = self.valid_loader
            self.model.eval()

        batches = len(dataloader)
        iter_loss = 0

        tk1 = tqdm(dataloader, total=batches)
        tk1.set_description(desc=f"{phase}")
        for i, (imgs, labels) in enumerate(tk1, 1):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            if phase == 'train':
                outs = self.model(imgs)
                loss = self.loss_fn(outs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            else:
                outs = self.model(imgs)
                loss = self.loss_fn(outs, labels)

            iter_loss += loss.item()

            tk1.set_postfix(lr=f"{self.optimizer.param_groups[0]['lr']:.4g}",
                            loss=f"{iter_loss/i:.4g}")
            
        iter_loss /= batches
        
        torch.cuda.empty_cache()
        gc.collect()

        return iter_loss
    

    def fit(self):
        best_loss = 10**9
        best_mAP = 0

        curr_loss_patience = 0
        curr_mAP_patience = 0

        save_dir = self.log_dir + '/model_' \
                + f'{self.uses}_fold_{self.fold}_of_{self.nb_splits}' \
                + f'_iou_{self.iou_threshold}.pth'

        tk0 = tqdm(range(self.epochs), total=self.epochs)
        for epoch in tk0:
            print(f"Epoch: {epoch+1} / {self.epochs}")
            train_loss = self.iteration('train')
            train_mAP = get_mAP(model=self.model,
                                dataloader=self.train_loader,
                                device=self.device,
                                confidence_threshold=self.confidence_threshold,
                                nms_threshold=self.nms_threshold,
                                iou_threshold=self.iou_threshold)
            save_log(self.log_dir, self.use_all, self.fold, self.nb_splits,
                     self.iou_threshold, phase='train',
                     epoch=epoch, loss=train_loss, score=train_mAP)

            with torch.inference_mode():
                valid_loss = self.iteration('valid')
                valid_mAP = get_mAP(model=self.model,
                                    dataloader=self.valid_loader,
                                    device=self.device,
                                    confidence_threshold=self.confidence_threshold,
                                    nms_threshold=self.nms_threshold,
                                    iou_threshold=self.iou_threshold)

            tk0.set_postfix(train_loss=f"{train_loss:.4g}",
                            train_mAP=f"{train_mAP:.4g} @ IOU = {self.iou_threshold}",
                            valid_loss=f"{valid_loss:.4g}",
                            valid_mAP=f"{valid_mAP:.4g} @ IOU = {self.iou_threshold}")
            save_log(self.log_dir, self.use_all, self.fold, self.nb_splits, 
                     self.iou_threshold, phase='valid', 
                     epoch=epoch, loss=valid_loss, score=valid_mAP)
            
            flag = False
            if valid_mAP > best_mAP:
                flag = True
                best_mAP = valid_mAP
                curr_mAP_patience = 0
                
                torch.save(self.model.state_dict(), save_dir)
            else:
                curr_mAP_patience += 1

            if valid_loss < best_loss:
                best_loss = valid_loss
                curr_loss_patience = 0
                if not flag:
                    torch.save(self.model.state_dict(), save_dir)
            else:
                curr_loss_patience += 1
            
            if self.early_stop and (curr_loss_patience > self.loss_patience
                                    and curr_mAP_patience > self.mAP_patience):
                clear_output()
                print(f"Model doesn't improved for"
                      f"loss: {curr_loss_patience} / mAP: {curr_mAP_patience} epochs."
                      f"Stop training.")

                print(f"Trained model save at {save_dir}")

                break
        else:
            clear_output()
            # torch.save({
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': self.optimizer.state_dict(),
            #     }, save_dir)
            torch.save(self.model.state_dict(), save_dir)

            print("Done!")
            print(f"Valid Loss: {valid_loss:.4g} | "
                f"Valid mAP: {valid_mAP:.4g} @ IOU: {self.iou_threshold}")
            print(f"Trained model save at {save_dir}")