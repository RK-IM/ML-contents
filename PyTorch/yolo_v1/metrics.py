import torch
from collections import Counter

from params import S, B, C


def intersection_over_union(box_preds, box_labels, box_format="corners"):
    """
    Calculates intersection over union

    Args:
        boxes_preds (torch.Tensor [batch_size, grid_size, grid_size, 4]): 
            Predictions of Bounding Boxes
        boxes_labels (torch.Tensor [batch_size, grid_size, grid_size, 4]): 
            Target position of Bounding Boxes
        box_format (str): 'midpoint' or 'corners', 
            correspond to (x, y, w, h) and (x1, y1, x2, y2)
    
    Returns:
        (torch.Tensor): Intersection over union for all batches
    """
    if box_format == "corners":
        box_preds_x1 = box_preds[..., 0]
        box_preds_y1 = box_preds[..., 1]
        box_preds_x2 = box_preds[..., 2]
        box_preds_y2 = box_preds[..., 3]
        box_labels_x1 = box_labels[..., 0]
        box_labels_y1 = box_labels[..., 1]
        box_labels_x2 = box_labels[..., 2]
        box_labels_y2 = box_labels[..., 3]

    elif box_format == "midpoint":
        box_preds_x1 = box_preds[..., 0] - box_preds[..., 2] / 2
        box_preds_y1 = box_preds[..., 1] - box_preds[..., 3] / 2
        box_preds_x2 = box_preds[..., 0] + box_preds[..., 2] / 2
        box_preds_y2 = box_preds[..., 1] + box_preds[..., 3] / 2
        box_labels_x1 = box_labels[..., 0] - box_labels[..., 2] / 2
        box_labels_y1 = box_labels[..., 1] - box_labels[..., 3] / 2
        box_labels_x2 = box_labels[..., 0] + box_labels[..., 2] / 2
        box_labels_y2 = box_labels[..., 1] + box_labels[..., 3] / 2

    else:
        raise ValueError('only "corners" or "midpoint" format are supported')
    
    x1 = torch.max(box_preds_x1, box_labels_x1)
    y1 = torch.max(box_preds_y1, box_labels_y1)
    x2 = torch.min(box_preds_x2, box_labels_x2)
    y2 = torch.min(box_preds_y2, box_labels_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area_preds = abs((box_preds_x2 - box_preds_x1)
                      * (box_preds_y2 - box_preds_y1))
    area_labels = abs((box_labels_x2 - box_labels_x1) 
                      * (box_labels_y2 - box_labels_y1))
    
    return intersection / (area_preds + area_labels - intersection + 1e-6)


def non_max_suppression(bboxes, 
                        confidence_threshold=0.3,
                        iou_threshold=0.5,
                        box_format='midpoint'):
    """
    Calculate IOU and drop boxes which has same class as reference boundary box
    and lower confidence. High IOU with same class can be interpreted  as 
    boundary boxes are predicting same object.
    
    Args:
        bboxes (list, [[class_id, confidence, coordinates(4)], [], ...]):
            boundary boxes.
        confidence_threshold (float): Boundary box with confidence higher than 
            this value will only selected. Defaults to 0.3.
        iou_threshold (float): If IOU between two boxes which has same class
            is higher than this value, one with lower confidence will be dropped.
            Defaults to 0.5.
        box_format (str): coordinate format of boundary box.
            'midpoint': x, y, width, height
            'corners': x_min, y_min, x_max, y_max
            Defaults to 'midpoint'

    Returns:
        (list, [[class_id, confidence_score, coordinates(4)], [], ...]):
    """
    bboxes = [box for box in bboxes if box[1] > confidence_threshold]
    bboxes = sorted(bboxes, key=lambda box: box[1])
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop()
        bboxes = [box for box in bboxes
                  if box[0] != chosen_box[0]
                      or intersection_over_union(
                        torch.tensor(chosen_box[2:]),
                        torch.tensor(box[2:]),
                        box_format=box_format
                    ) < iou_threshold
            ]
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms


def mean_average_precision(pred_boxes,
                           true_boxes,
                           iou_threshold=0.5,
                           box_format="midpoint",
                           num_class=20):
    """
    Calculate PRAUC for each class and average them.

    Args:
        pred_boxes (list, [index, class_id, confidence, coordinates(4), [], ...]):
            Predicted boundary box. Index is to avoid redundant counting.
        true_boxes (list, [index, class_id, confidence, coordinates(4), [], ...]):
            Ground truth boundary box.
        iou_threshold (float): Select pred_box which has higher iou than
            this value respects to ground truth boundary box.
            Defaults to 0.5.
        box_format (str): coordinate format of boundary box.
            'midpoint': x, y, width, height
            'corners': x_min, y_min, x_max, y_max
            Defaults to 'midpoint'
        num_class (int): Number of unique labels. Defaults to 20(pascal VOC classes).

    Returns:
        (torch.Tensor(float)): mean average precision
    """
    average_precision = []

    # individual classes
    for c in range(num_class):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes:
            if detection[1] == c: # same class
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # save image index and number of objects included
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # Make Zeros that size is the length of ground truths
        # to check if an object in that image has been counted.
        for k, val in amount_bboxes.items():
            amount_bboxes[k] = torch.zeros(val)

        # Sort detected boxes by confidence which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue
        
        for detection_idx, detection in enumerate(detections):
            # For caculates the true positive and false positive
            # of certain image, select ground truth image
            # that has same image index of detection.
            ground_truth_img = [bbox for bbox in ground_truths
                                if bbox[0] == detection[0]]
            
            best_iou = 0
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),                          
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )
                if iou > best_iou:
                    best_iou = iou
                    gt_idx = idx

            if best_iou > iou_threshold:
                # check if gt_idx has been counted
                if amount_bboxes[detection[0]][gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        # To calculate PRAUC, cumulate TP and FP
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        precision = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        recall = TP_cumsum / (total_true_bboxes + 1e-6)
        precision = torch.cat([torch.tensor([1]), precision])
        recall = torch.cat([torch.tensor([0]), recall])

        average_precision.append(torch.trapz(y=precision, x=recall))
    
    return sum(average_precision) / len(average_precision)
        

def out2cellbox(outs, S=S, B=B, C=C):
    """
    Convert model outputs to boundary boxes which has coordinates 
    relative to the entire image.

    Args:
        outs (torch.Tensor[batch_size, S * S * (C + B*5)])
    
    Returns:
        (torch.Tensor[batch_size, S, S, class, confidence, coordinates(4))])
    """
    batch_size = outs.shape[0]
    outs = outs.cpu()
    outs = outs.reshape(batch_size, S, S, C + B*5)

    # Select predicted/label class for each cells
    class_id = outs[..., :20].argmax(-1).unsqueeze(-1)

    # Select higher confidence between two boundary boxes.
    # and select boundary box coordinates info based on confidence.
    scores = torch.cat([outs[..., 20:21], outs[..., 25:26]], dim=-1)
    # confidence and box indices
    confidence, best_box = torch.max(scores, dim=-1)
    confidence = confidence.unsqueeze(-1).clip(0, 1)
    best_box = best_box.unsqueeze(-1)

    bboxes1 = outs[..., 21:25]
    bboxes2 = outs[..., 26:30]

    # box coordinates
    best_boxes = bboxes1 * (1 - best_box) \
        + bboxes2 * (best_box)
    
    # Create cell indices to calculate relative coordinates
    # responeding to entire image
    xcell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    ycell_indices = xcell_indices.permute(0, 2, 1, 3)

    # Calculate relative box coordinates, width, heights using cell indicies
    x_center = 1/7 * (best_boxes[..., :1] + xcell_indices)
    y_center = 1/7 * (best_boxes[..., 1:2] + ycell_indices)
    width_height = 1/7 * best_boxes[..., 2:4]

    # concatenate each values of coordinates info
    converted_bboxes = torch.cat([x_center, y_center, width_height], dim=-1)

    # concatenate results (class, confidence, coordinates(4))
    gridization_outs = torch.cat([class_id, confidence, converted_bboxes], dim=-1)

    return gridization_outs


def out2boxlist(outs):
    """
    outs (torch.Tensor[batch_size, S * S * (C + B*5)])
    gridization_outs (torch.Tensor[[class, confidence, coordinates(4)], [], ...])
    """
    gridization_outs = out2cellbox(outs).reshape(outs.shape[0], S*S, -1)

    all_bboxes = [] # bbox info for batches
    for batch_idx in range(outs.shape[0]):
        bboxes = [] # individual grid cell
        for grid_idx in range(S*S):
            bboxes.append([box.item()
                           for box in gridization_outs[batch_idx, grid_idx, :]])
        all_bboxes.append(bboxes)
    
    return all_bboxes