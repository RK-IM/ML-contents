import torch
import torch.nn as nn

from metrics import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        """
        Constructor.

        Args:
            S (int): Grid size. Defaults to 7.
            B (int): Number of boxes used for prediction. Defaults to 2.
            C (int): Classes in Pascal VOC. Defaults to 20.
        """
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5


    def forward(self, pred, true):
        # [class(0~19), 
        # confidence(20), coord(21~24), bbox 1
        # confidence(25), coord(26~29)] bbox 2
        pred = pred.reshape(-1, self.S, self.S, self.C + self.B*5)

        iou_pred1 = intersection_over_union(pred[..., 21:25],
                                            true[..., 21:25], 
                                            box_format="midpoint")
        iou_pred2 = intersection_over_union(pred[..., 26:30], 
                                            true[..., 21:25], 
                                            box_format="midpoint")
        
        ious = torch.cat([iou_pred1.unsqueeze(0), 
                          iou_pred2.unsqueeze(0)], 
                         dim=0)
        # "responsible" predictor for the ground truth box
        _, best_box = torch.max(ious, dim=0) # 0: pred1, 1: pred2
        best_box = best_box.unsqueeze(-1) # dimension match
        # Identity matrix which denotes the object exist in certain cell.
        box_exist = true[..., 20].unsqueeze(-1) # I_obj


        ### 1., 2. Box coordinates, first and second line of Loss equation (3)
        box_preds = box_exist * ( # select responsible predictor
            # select pred1 -> best_box = 0
            best_box * pred[..., 26:30]
            + (1 - best_box) * pred[..., 21:25]
        )

        box_true = box_exist * true[..., 21:25] # == true[..., 21:25]

        # Apply square root to width and height
        box_preds[..., 2:4] = torch.sign(box_preds[..., 2:4]) \
            * torch.sqrt(torch.abs(box_preds[..., 2:4] + 1e-6))
        box_true[..., 2:4] = torch.sqrt(box_true[..., 2:4])

        box_loss = self.mse(
            # [(batch_size * S * S), coordinats(4)]
            torch.flatten(box_preds, end_dim=-2), 
            torch.flatten(box_true, end_dim=-2)
        )


        ### 3. Object contains, third line of Loss equation(3)
        obj_preds = (
            best_box * pred[..., 25:26]
            + (1 - best_box) * pred[..., 20:21]
        )


        obj_loss = self.mse(
            # [(batch_size * S * S), 1]
            torch.flatten(box_exist * obj_preds, end_dim=-2),
            torch.flatten(box_exist * true[..., 20:21], end_dim=-2)
        )


        ### 4. No Object, fourth line
        # [(batch_size * S * S), 1]
        no_obj_loss = self.mse(
            torch.flatten((1 - box_exist) * pred[..., 20:21], end_dim=-2),
            torch.flatten((1 - box_exist) * true[..., 20:21], end_dim=-2)
        )
        
        no_obj_loss += self.mse(
            torch.flatten((1 - box_exist) * pred[..., 25:26], end_dim=-2),
            torch.flatten((1 - box_exist) * true[..., 20:21], end_dim=-2)
        )


        ### 5. Class loss, fifth line
        class_loss = self.mse(
            # [(batch_size * S * S), classes(20)]
            torch.flatten(box_exist * pred[..., :20], end_dim=-2),
            torch.flatten(box_exist * true[..., :20], end_dim=-2),
        )

        # overall
        loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * no_obj_loss
            + class_loss
        )
        return loss
