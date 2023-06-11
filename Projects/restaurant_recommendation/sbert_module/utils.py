from config import CFG

from pathlib import Path

import pandas as pd

import torch
from sklearn.metrics import auc
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import F1Score, AUROC, PrecisionRecallCurve

def calculate_metrics(preds: torch.tensor,
                      true: torch.tensor):
    f1 = F1Score(task='multilabel', num_labels=15, average=None)
    auroc = AUROC(task='multilabel', num_labels=15, average=None)
    pr_curve = PrecisionRecallCurve(task='multilabel', num_labels=15, average=None)

    f1_score = f1(preds, true).numpy()
    auroc_score = auroc(preds, true).numpy()

    precision, recall, _ = pr_curve(preds, true)
    prauc_score = [auc(r, p) for r, p in zip(recall, precision)]

    results = pd.DataFrame([f1_score, auroc_score, prauc_score],
                           columns=CFG.CLASS_NAMES).T
    results = results.rename(columns={0: 'F1 score', 1: 'ROC AUC', 2: 'PR AUC'})
    # results = results.style.format('{:.3f}').bar(vmin=0, vmax=1, color='#cde5ae', width=40,
    #                                              props="width: 60px;")

    return results

def save_model(model: torch.nn.Module, 
               model_path: str,
               model_name: str):
    model_path = Path(model_path)
    if not model_path.is_dir():
        model_path.mkdir(exist_ok=True, parents=True)
    
    assert model_name.endswith('.pth') or model_name.endswith('.pt'), "model_name should ends with '.pt' or' .pth'"
    torch.save(model.state_dict(), model_path / model_name)
    print(f"[INFO]Model saved! {model_path}/{model_name}")