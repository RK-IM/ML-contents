from config import CFG
from data_setup import get_data, ClassifierDataset
import sbert_trainer
from model_builder import load_sbert, SimpleMLC
from utils import save_model
import engine

import os
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set parser
parser = argparse.ArgumentParser()
parser.add_argument('--update',
                    help="If True, update inference model to current updated model, Default is True",
                    action='store_true')

args = parser.parse_args()

# update sbert
sbert_save_date = sbert_trainer.sbert_train(
    train_csv_file=CFG.UPDATE_CSV_FILE)

# load sbert which learn right before
sbert_model = load_sbert(CFG.SBERT_MODEL_PATH, 
                         sbert_save_date)

# get feedback data
update_df = get_data(CFG.DATA_PATH, CFG.UPDATE_CSV_FILE, column='review')
X = update_df['review'].tolist()
y = update_df.loc[:, update_df.dtypes==int].to_numpy()

# encode feedback data and create dataloader for classifier
X = sbert_model.encode(X)
dataset = ClassifierDataset(X, y)
dataloader = DataLoader(dataset,
                        batch_size=CFG.BATCH_SIZE,
                        shuffle=True)

# load previous classifier
classifier = SimpleMLC(n_classes=CFG.N_CLASSES).to(device)

if device == 'cpu':
    classifier.load_state_dict(
        torch.load(f'{CFG.CLASSIFIER_MODEL_PATH}/{CFG.CLASSIFIER_MODEL_DATE}/{CFG.CLASSIFIER_MODEL_FILE}.pth',
        map_location=torch.device('cpu')))
else:
    classifier.load_state_dict(
        torch.load(f'{CFG.CLASSIFIER_MODEL_PATH}/{CFG.CLASSIFIER_MODEL_DATE}/{CFG.CLASSIFIER_MODEL_FILE}.pth'))

# set functions
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(params=classifier.parameters(),
                              lr=1e-7, weight_decay=0.99)

# update classifier
update_loss, update_resuls = engine.train_step(
    model=classifier,
    dataloader=dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    device=device)

# save model
classifier_model_name = f"{CFG.CLASSIFIER_MODEL_NEW_FILE}"
classifier_save_path = f"{CFG.CLASSIFIER_MODEL_PATH}/{sbert_save_date}"

save_model(classifier, classifier_save_path, f"{classifier_model_name}.pth")

# write log
log_path = Path('./runs')
if not log_path.is_dir():
    log_path.mkdir(exist_ok=True, parents=True)

writer = SummaryWriter(log_dir=log_path/f"{sbert_save_date}")
# loss
writer.add_scalars(main_tag="Loss",
                   tag_scalar_dict={'train_loss': update_loss},
                   global_step=1)

# metrics
writer.add_scalars(main_tag="F1_score",
                   tag_scalar_dict=update_resuls.loc[:, 'F1 score'].to_dict(),
                   global_step=1)
writer.add_scalars(main_tag="ROC_AUC",
                   tag_scalar_dict=update_resuls.loc[:, 'ROC AUC'].to_dict(),
                   global_step=1)
writer.add_scalars(main_tag="PR_AUC",
                   tag_scalar_dict=update_resuls.loc[:, 'PR AUC'].to_dict(),
                   global_step=1)

# update config and remove previous model
def change_cfg(config_file, attribute, value):
    with open(config_file, 'r') as f:
        cfg = f.readlines()
    with open(config_file, 'w') as f:
        for line in cfg:
            if line.__contains__(attribute):
                f.writelines(f"    {attribute} = '{value}'\n")
            else:
                f.write(line)

if (args.update == True) and (len(os.listdir(CFG.SBERT_MODEL_PATH)) > 2):
    shutil.rmtree(f"{CFG.SBERT_MODEL_PATH}/{CFG.SBERT_MODEL_FOLDER}")
    shutil.rmtree(f"{CFG.CLASSIFIER_MODEL_PATH}/{CFG.CLASSIFIER_MODEL_DATE}")
    change_cfg('config.py', 'SBERT_MODEL_FOLDER', sbert_save_date)
    change_cfg('config.py', 'CLASSIFIER_MODEL_DATE', sbert_save_date)
    change_cfg('config.py', 'CLASSIFIER_MODEL_FILE', classifier_model_name)

    print("[INFO] Model updated! Config file modified.")