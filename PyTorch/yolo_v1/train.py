import torch

from tqdm import tqdm
from IPython.display import clear_output

from metrics import (
    non_max_suppression,
    mean_average_precision,
    out2boxlist,
)
from utils import save_log


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

    
    def get_mAP(self, dataloader, box_format="midpoint"):
        """
        Calculate mean average precsion using every outputs of model.

        Args:
            dataloader (torch dataloader): Data loaders that contains 
                data to calculate mAP.
            box_format (str): 'midpoint' or 'corners', 
                correspond to (x, y, w, h) and (xmin, ymin, xmax, ymax)
        
        Return:
            mAP (float)
        """
        all_true_bboxes = []
        all_pred_bboxes = []

        self.model.eval()
        idx = 0
        with torch.inference_mode():
            for imgs, lbls in dataloader:
                batch_size = imgs.shape[0]
                imgs, lbls = imgs.to(self.device), lbls.to(self.device)

                outs = self.model(imgs)
                true_bboxes = out2boxlist(lbls)
                pred_bboxes = out2boxlist(outs)

                for i in range(batch_size):
                    nms_bboxes = non_max_suppression(
                        pred_bboxes[i],
                        confidence_threshold=self.confidence_threshold,
                        iou_threshold=self.nms_threshold,
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
                                    iou_threshold=self.iou_threshold,
                                    box_format=box_format)
        
        return mAP.item()
    

    def iteration(self):
        train_loss = 0
        valid_loss = 0

        self.model.train()
        for i, (imgs, labels) in enumerate(self.train_loader, 1):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            outs = self.model(imgs)
            loss = self.loss_fn(outs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            train_loss += loss.item()
        
        self.model.eval()
        with torch.inference_mode():
            for i, (imgs, labels) in enumerate(self.valid_loader, 1):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outs = self.model(imgs)
                loss = self.loss_fn(outs, labels)
                valid_loss += loss.item()
        
        train_loss /= len(self.train_loader)
        valid_loss /= len(self.valid_loader)

        return train_loss, valid_loss


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
            train_loss, valid_loss = self.iteration()
            train_mAP = self.get_mAP(self.train_loader)
            valid_mAP = self.get_mAP(self.valid_loader)

            save_log(self.log_dir, self.use_all, self.fold, self.nb_splits,
                     self.iou_threshold, phase='train',
                     epoch=epoch, loss=train_loss, score=train_mAP)
            save_log(self.log_dir, self.use_all, self.fold, self.nb_splits, 
                     self.iou_threshold, phase='valid', 
                     epoch=epoch, loss=valid_loss, score=valid_mAP)

            tk0.set_postfix(train_loss=f"{train_loss:.4g}",
                            train_mAP=f"{train_mAP:.4g} @IoU: {self.iou_threshold}",
                            valid_loss=f"{valid_loss:.4g}",
                            valid_mAP=f"{valid_mAP:.4g} @IoU: {self.iou_threshold}")
            
            flag = False
            if valid_mAP > best_mAP:
                flag = True
                best_mAP = valid_mAP
                curr_mAP_patience = 0
                if best_mAP > 0.5:
                    torch.save(self.model.state_dict(), save_dir)

            else:
                curr_mAP_patience += 1

            if valid_loss < best_loss:
                best_loss = valid_loss
                curr_loss_patience = 0
                if not flag and best_loss < 100:
                    torch.save(self.model.state_dict(), save_dir)
            else:
                curr_loss_patience += 1
            
            if self.early_stop and (curr_loss_patience > self.loss_patience
                                    and curr_mAP_patience > self.mAP_patience):
                clear_output()
                print(f"Model doesn't improved for "
                      f"loss: {curr_loss_patience} / mAP: {curr_mAP_patience} epochs. "
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