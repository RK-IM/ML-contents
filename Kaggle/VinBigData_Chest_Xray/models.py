import sys

sys.path.append("../timm-efficientdet-pytorch")

import os
import numpy as np
import pandas as pd

import timm
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ensemble_boxes import weighted_boxes_fusion
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.config import get_efficientdet_config
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.efficientdet import HeadNet

from dataset import pred_transforms
from params import TEST_PATH


# ------------#
# Classifier #
# ------------#
class XrayClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name="tf_efficientnet_b0",
        pretrained=True,
        num_classes=1,
        lr=1e-3,
        weight_decay=1e-5,
        gamma=0.99,
        max_epochs=10,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.max_epochs = max_epochs

        self.model = self.get_model()
        self.metrics = torch.nn.ModuleDict(  # without wrapping with ModuleDict,
            {
                "accuracy": Accuracy(
                    task="binary"
                ),  # torchmetric throw device mismatch error
                "auroc": AUROC(task="binary"),
            }
        )
        self.step_outputs = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        self.epoch_outputs = {
            "train_preds": [],
            "train_labels": [],
            "val_preds": [],
            "val_labels": [],
        }
        self.predict_step_outputs = []

    def forward(self, images):  # inference action
        print(images)
        return torch.sigmoid(self.model(images))

    # Step
    def _common_step(self, batch, stage):
        images, labels = batch
        outs = self.model(images).squeeze()
        preds = torch.sigmoid(outs)

        loss = F.binary_cross_entropy_with_logits(outs, labels.float())

        self.step_outputs[f"{stage}_loss"].append(loss)
        self.step_outputs[f"{stage}_accuracy"].append(
            self.metrics["accuracy"](preds, labels)
        )

        self.epoch_outputs[f"{stage}_preds"].extend(preds)
        self.epoch_outputs[f"{stage}_labels"].extend(labels)

        return loss

    def training_step(self, batch, batch_idx):  # model output sigmoid not applied
        return self._common_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, stage="val")

    # Epoch
    def _common_epoch_end(self, stage):
        self.epoch_outputs[f"{stage}_preds"] = torch.stack(
            self.epoch_outputs[f"{stage}_preds"]
        )
        self.epoch_outputs[f"{stage}_labels"] = torch.stack(
            self.epoch_outputs[f"{stage}_labels"]
        )

        loss = torch.mean(
            torch.tensor([loss for loss in self.step_outputs[f"{stage}_loss"]])
        )
        accuracy = torch.mean(
            torch.tensor([acc for acc in self.step_outputs[f"{stage}_accuracy"]])
        )
        auroc = self.metrics["auroc"](
            self.epoch_outputs[f"{stage}_preds"], self.epoch_outputs[f"{stage}_labels"]
        )

        for step_key, epoch_key in zip(["loss", "accuracy"], ["preds", "labels"]):
            self.step_outputs[f"{stage}_{step_key}"].clear()
            self.epoch_outputs[f"{stage}_{epoch_key}"] = []

        self.log_dict(
            {
                f"{stage}_loss": loss,
                f"{stage}_accuracy": accuracy,
                f"{stage}_auroc": auroc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_train_epoch_end(self):
        return self._common_epoch_end(stage="train")

    def on_validation_epoch_end(self):
        return self._common_epoch_end(stage="val")

    # Predict
    def predict_step(self, batch, batch_idx):
        images, _, image_name = batch
        outs = torch.sigmoid(self.model(images))
        self.predict_step_outputs.append([outs, image_name])
        return outs

    def on_predict_epoch_end(self):
        results = self.predict_step_outputs
        ids = []
        vals = []
        for i in range(len(results)):
            ids.extend(results[i][1])
            vals.extend(results[i][0].cpu())

        results = dict(zip(ids, vals))
        results = {k.split(".")[0]: v.item() for k, v in results.items()}
        self.predict_result = results

    # Model, Optimizer
    def get_model(self):
        return timm.create_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]


# ----------#
# Detector #
# ----------#
class XrayDetector(pl.LightningModule):
    def __init__(
        self,
        num_classes=14,
        image_size=512,
        prediction_confidence_threshold=0.3,
        learning_rate=2e-4,
        weight_decay=1e-5,
        gamma=0.99,
        wbf_iou_threshold=0.4,
        inference_transforms=pred_transforms(),
        architecture="tf_efficientnet_b0",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = image_size
        self.architecture = architecture

        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_transforms = inference_transforms

        self.model = self.get_model(
            num_classes=self.num_classes,
            image_size=self.img_size,
            architecture=self.architecture,
        )

        self.validation_step_outputs = []
        self.predict_step_outputs = []

    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, annotations, _ = batch
        losses = self.model(images, annotations)  # loss dict

        self.log_dict(
            {
                "train_loss": losses["loss"],
                "train_class_loss": losses["class_loss"],
                "train_box_loss": losses["box_loss"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=tuple,
        )

        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        images, annotations, targets = batch
        outputs = self.model(images, annotations)  # loss dict + detection
        detections = outputs["detections"]  # [bboxes[4], score, classes]

        batch_detections = {
            "predictions": detections,
            "targets": targets,
        }

        self.log_dict(
            {
                "valid_loss": outputs["loss"],
                "valid_class_loss": outputs["class_loss"],
                "valid_box_loss": outputs["box_loss"],
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        outs = {"loss": outputs["loss"], "batch_predictions": batch_detections}

        self.validation_step_outputs.append(outs)

        return outs

    def on_validation_epoch_end(self):
        (
            pred_labels,
            pred_bboxes,
            pred_confidences,
            targets,
        ) = self.aggregate_val_outputs()
        pred_bboxes = list(map(np.array, pred_bboxes))
        pred_bboxes = [b[:, [1, 0, 3, 2]] for b in pred_bboxes]

        mAP = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", iou_thresholds=[0.4]
        )
        preds = [
            {
                "boxes": torch.tensor(b),
                "scores": torch.tensor(c),
                "labels": torch.tensor(l).to(int),
            }
            for b, c, l in zip(pred_bboxes, pred_confidences, pred_labels)
        ]
        target = [
            {"boxes": t["bboxes"].cpu(), "labels": t["labels"].cpu().to(int)}
            for t in targets
        ]

        self.val_final = [preds, target]

        mAP.update(preds, target)
        map_40 = mAP.compute()["map"].item()
        self.log(
            "mAP @[IoU=0.40]",
            map_40,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def predict_step(self, batch, batch_idx):
        images, image_sizes, image_names = batch
        if images.ndim == 3:
            images = images.unsqueeze(0)

        outs = self._run_inference(images, image_sizes)
        self.predict_step_outputs.append(outs)

        return outs

    def on_predict_epoch_end(self):
        coords = []
        confidences = []
        labels = []
        for out in self.predict_step_outputs:
            coords.extend(out[0])
            confidences.extend(out[1])
            labels.extend(out[2])
        image_ids = os.listdir(TEST_PATH)

        self.predict_result = pd.DataFrame(
            {
                "image_id": image_ids,
                "labels": labels,
                "confidences": confidences,
                "box_coordinates": coords,
            }
        )

    def predict_sample(self, image):
        if len(image.shape) == 2:
            image = image = np.stack([image] * 3).transpose(1, 2, 0)
        image_size = image.shape[:2]
        image_tfmed = self.inference_tfms(image=image)["image"]
        image_tfmed = image_tfmed.unsqueeze(0)

        return self._run_inference(image_tfmed, image_size)

    def _run_inference(self, images, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(num_images=images.shape[0])
        outputs = self.model(images.to(self.device), dummy_targets)

        # detections: [batch_size, n_bboxes, [bboxes[4], score, classes]]
        detections = outputs["detections"]

        pred_bboxes, pred_confidences, pred_labels = self.post_process_detections(
            detections
        )

        scaled_bboxes = self.__rescale_bboxes(pred_bboxes, image_sizes)

        return scaled_bboxes, pred_confidences, pred_labels

    def _create_dummy_inference_targets(self, num_images):
        return {
            "bbox": [
                torch.tensor([[0, 0, 0, 0]], device=self.device, dtype=torch.float)
                for _ in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for _ in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(self._post_process_single_batch(detections[i]))
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

        return {
            "boxes": boxes[candidates],  # [batch_size, n_boxes, coords[4]]
            "scores": scores[candidates],  # [batch_size, confidence_score]
            "classes": classes[candidates],
        }  # [batch_size, classes]

    def apply_wbf(self, predictions, image_size, iou_thr=0.4, skip_box_thr=0.0001):
        # ZFTurbo's wbf expects normalized box coordinates
        wbf_boxes = []
        wbf_scores = []
        wbf_labels = []
        for prediction in predictions:
            boxes_list = prediction["boxes"] / image_size
            if len(boxes_list) == 0:
                wbf_boxes.append(np.zeros((1, 4)),)
                wbf_scores.append(np.array([0.0]))
                wbf_labels.append(np.array([0.0]))
                continue

            wbf_box, wbf_score, wbf_label = weighted_boxes_fusion(
                [boxes_list],
                [prediction["scores"]],
                [prediction["classes"]],
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
            wbf_box = (wbf_box * image_size).round()

            wbf_boxes.append(wbf_box)
            wbf_scores.append(wbf_score)
            wbf_labels.append(wbf_label)

        return wbf_boxes, wbf_scores, wbf_labels

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        if isinstance(image_sizes, list):
            image_sizes = torch.stack(image_sizes).T
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            if not isinstance(img_dims, torch.Tensor):
                im_h, im_w = img_dims
            else:
                im_h, im_w = img_dims.cpu()

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes)
                        * [
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)
        return scaled_bboxes

    def aggregate_val_outputs(self):
        detections = torch.cat(
            [
                output["batch_predictions"]["predictions"]
                for output in self.validation_step_outputs
            ]
        ).cpu()

        targets = []
        for output in self.validation_step_outputs:
            batch_predictions = output["batch_predictions"]
            targets.extend(batch_predictions["targets"])

        pred_bboxes, pred_confidences, pred_labels = self.post_process_detections(
            detections
        )

        return pred_labels, pred_bboxes, pred_confidences, targets

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]

    def get_model(
        self, num_classes=14, image_size=512, architecture="tf_efficientnet_b0"
    ):
        # New element to efficientdet_mode_param_dict
        # backbone name should be in timm.list_models()
        efficientdet_model_param_dict[architecture] = dict(
            name=architecture,
            backbone_name=architecture,
            backbone_args=dict(),
            num_classes=num_classes,
            url="",
        )

        # get effdet config and overwirte with above efficientdet_model_param's model_name
        config = get_efficientdet_config(architecture)
        config.update({"num_classes": num_classes})
        config.update({"image_size": (image_size, image_size)})

        # define effdet model by created config
        net = EfficientDet(config, pretrained_backbone=True)
        # HeadNet is basically a custom head that takes the final outputs of the BiFPN network
        # and either returns a class or bounding box coordinates
        net.class_net = HeadNet(config, num_outputs=config.num_classes)
        # @train: return losses, else return detections
        return DetBenchTrain(net)
