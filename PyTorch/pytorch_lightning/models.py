import sys
sys.path.append("..")

import os
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from ensemble_boxes import weighted_boxes_fusion

from dataset import get_test_transforms
from params import TEST_PATH


def create_model(num_classes=1, image_size=512, architecture="tf_efficientnetv2_l"):
    efficientdet_model_param_dict[architecture] = dict(
        name=architecture,
        backbone_name=architecture,
        backbone_args=dict(drop_path_rate=0.2),
        num_classes=num_classes,
        url="",
    )

    config = get_efficientdet_config(architecture)
    config.update({"num_classes": num_classes})
    config.update({"image_size": (image_size, image_size)})

    print(config)

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    
    return DetBenchTrain(net)


class EfficientDetModel(pl.LightningModule):
    def __init__(
        self,
        num_classes=1,
        img_size=512,
        prediction_confidence_threshold=0.3,
        learning_rate=2e-4,
        wbf_iou_threshold=0.4,
        inference_transforms=get_test_transforms(target_img_size=512),
        model_architecture="tf_efficientnetv2_l",
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes, img_size, architecture=model_architecture, 
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms

        self.validation_step_outputs = []
        self.predict_step_outputs = []

    def forward(self, images, targets):
        return self.model(images, targets)
    

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch

        losses = self.model(images, annotations)

        logging_losses = {
            "class_loss": losses["class_loss"].detach(),
            "box_loss": losses["box_loss"].detach(),
        }

        self.log("train_loss", losses["loss"], 
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_class_loss", logging_losses["class_loss"],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_box_loss", logging_losses["box_loss"],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return losses["loss"]
    

    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch

        outputs = self.model(images, annotations)

        detections = outputs["detections"] # bboxes[4], score, classes

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log("valid_loss", outputs["loss"],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid_class_loss", logging_losses["class_loss"],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid_box_loss", logging_losses["box_loss"],
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        outs = {"loss": outputs["loss"],
                "batch_predictions": batch_predictions}

        self.validation_step_outputs.append(outs)

        return outs


    def on_validation_epoch_end(self):
        pred_labels, _, pred_bboxes, pred_confidences, targets = self.aggregate_val_outputs()
        pred_bboxes = list(map(np.array, pred_bboxes))
        pred_bboxes = [b[:, [1, 0, 3, 2]] for b in pred_bboxes]

        mAP = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", iou_thresholds=[0.4])
        preds = [{"boxes": torch.tensor(b), "scores": torch.tensor(c), "labels": torch.tensor(l).to(int)}
                 for b, c, l in zip(pred_bboxes, pred_confidences, pred_labels)]
        target = [{"boxes": t["bboxes"].cpu(), "labels": t["labels"].cpu().to(int)} for t in targets]

        mAP.update(preds, target)
        map_40 = mAP.compute()["map"].item()
        self.log("mAP @[IoU=0.40]", map_40, 
                 on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


    def predict_step(self, batch, batch_idx):
        images, image_sizes, image_ids = batch
        if images.ndim == 3:
            images = images.unsqueeze(0)
        
        outs = self._run_inference(images, image_sizes)
        self.predict_step_outputs.append(outs)

        return outs


    def on_predict_epoch_end(self):
        coords = []
        confidences = []
        for out in self.predict_step_outputs:
            coords.extend(out[0])
            confidences.extend(out[1])
        image_ids = os.listdir(TEST_PATH)

        self.predict_result = pd.DataFrame(
            {"image_id": image_ids,
             "box_coordinates": coords,
             "confidences": confidences}
        )
    

    def predict_sample(self, image):
        image_size = image.shape[:2]
        image_tfmed = self.inference_tfms(image=image)["image"]
        image_tfmed = image_tfmed.unsqueeze(0)

        return self._run_inference(image_tfmed, image_size)
    

    def _run_inference(self, images, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(num_images=images.shape[0])
        outputs = self.model(images.to(self.device), dummy_targets)

        # detections: [batch_size, n_bboxes, [bboxes[4], score, classes]]
        detections = outputs["detections"] 

        pred_bboxes, pred_confidences, pred_labels = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(pred_bboxes, image_sizes)

        return scaled_bboxes, pred_confidences, pred_labels

    
    def _create_dummy_inference_targets(self, num_images):
        return {
            "bbox": [torch.tensor([[0, 0, 0, 0]], device=self.device, dtype=torch.float)
                     for _ in range(num_images)],
            "cls": [torch.tensor([1.], device=self.device)
                    for _ in range(num_images)],
            "img_size": torch.tensor([(self.img_size, self.img_size)] * num_images,
                                     device=self.device).float(),
            "img_scale": torch.ones(num_images, device=self.device).float()
        }
    

    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._post_process_single_batch(detections[i])
            )
        # wbf by single image [torch.tensor(pred), torch.tensor(pred), ...]
        pred_bboxes, pred_confidences, pred_labels = self.apply_wbf(
            predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold
        )
        return pred_bboxes, pred_confidences, pred_labels
    

    def _post_process_single_batch(self, detections):
        boxes = detections.detach().cpu()[:, :4]
        scores = detections.detach().cpu()[:, 4]
        classes = detections.detach().cpu()[:, 5]
        candidates = torch.where(scores > self.prediction_confidence_threshold)[0]

        return {"boxes": boxes[candidates], # [batch_size, n_boxes, coords[4]]
                "scores": scores[candidates], # [batch_size, confidence_score]
                "classes": classes[candidates]} # [batch_size, classes]


    def apply_wbf(self, predictions, image_size, iou_thr=0.4, skip_box_thr=0.0001):
        # ZFTurbo's wbf expects normalized box coordinates
        wbf_boxes = []
        wbf_scores = []
        wbf_labels = []
        for prediction in predictions:
            boxes_list = prediction["boxes"] / image_size
            if len(boxes_list) == 0:
                wbf_boxes.append(np.zeros((1, 4)),)
                wbf_scores.append(np.array([0.]))
                wbf_labels.append(np.array([0.]))
                continue
            
            wbf_box, wbf_score, wbf_label = weighted_boxes_fusion(
                [boxes_list], 
                [prediction["scores"]], 
                [prediction["classes"]],
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr)
            wbf_box = (wbf_box * image_size).round()

            wbf_boxes.append(wbf_box)
            wbf_scores.append(wbf_score)
            wbf_labels.append(wbf_label)

        return wbf_boxes, wbf_scores, wbf_labels
    

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        if isinstance(image_sizes, list):
            image_sizes = torch.stack(image_sizes).T
        else:
            image_sizes = [image_sizes]
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            if not isinstance(img_dims, torch.Tensor):
                im_h, im_w = img_dims
            else:
                im_h, im_w = img_dims.cpu()

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (np.array(bboxes) * [im_w / self.img_size,
                                         im_h / self.img_size,
                                         im_w / self.img_size,
                                         im_h / self.img_size,]).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)
        return scaled_bboxes


    def aggregate_val_outputs(self):
        detections = torch.cat(
            [output["batch_predictions"]["predictions"]
             for output in self.validation_step_outputs]
        ).cpu()

        image_ids = []
        targets = []
        for output in self.validation_step_outputs:
            batch_predictions = output["batch_predictions"]
            image_ids.extend(batch_predictions["image_ids"])
            targets.extend(batch_predictions["targets"])
        
        pred_bboxes, pred_confidences, pred_labels = self.post_process_detections(detections)

        return pred_labels, image_ids, pred_bboxes, pred_confidences, targets