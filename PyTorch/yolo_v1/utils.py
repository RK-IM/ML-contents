import torch

def intersection_over_union(box_preds, box_labels, format="corners"):

    if format == "corners":
        box_preds_x1 = box_preds[..., 0]
        box_preds_y1 = box_preds[..., 1]
        box_preds_x2 = box_preds[..., 2]
        box_preds_y2 = box_preds[..., 3]
        box_labels_x1 = box_labels[..., 0]
        box_labels_y1 = box_labels[..., 1]
        box_labels_x2 = box_labels[..., 2]
        box_labels_y2 = box_labels[..., 3]

    elif format == "midpoints":
        box_preds_x1 = box_preds[..., 0] - box_preds[..., 2] / 2
        box_preds_y1 = box_preds[..., 1] - box_preds[..., 3] / 2
        box_preds_x2 = box_preds[..., 0] + box_preds[..., 2] / 2
        box_preds_y2 = box_preds[..., 1] + box_preds[..., 3] / 2
        box_labels_x1 = box_labels[..., 0] - box_labels[..., 2] / 2
        box_labels_y1 = box_labels[..., 1] - box_labels[..., 3] / 2
        box_labels_x2 = box_labels[..., 0] + box_labels[..., 2] / 2
        box_labels_y2 = box_labels[..., 1] + box_labels[..., 3] / 2

    else:
        raise ValueError('only "corners" or "midpoints" format are supported')
    
    x1 = torch.max(box_preds_x1, box_labels_x1)
    y1 = torch.max(box_preds_y1, box_labels_y1)
    x2 = torch.min(box_preds_x2, box_labels_x2)
    y2 = torch.min(box_preds_y2, box_labels_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    area_preds = abs((box_preds_x2 - box_preds_x1)
                      * (box_preds_y2 - box_preds_y1))
    area_labels = abs((box_labels_x2 - box_labels_x1) 
                      * (box_labels_y2 - box_labels_y1))
    
    return intersection / (area_preds + area_labels + 1e-6)


def non_maximum_suppression():
    pass
