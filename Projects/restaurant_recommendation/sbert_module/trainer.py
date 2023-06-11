from config import CFG
from data_setup import get_data, data_split, ClassifierDataset
from model_builder import load_sbert, SimpleMLC
from engine import train
from utils import save_model

import time
import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load sbert
timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Load SBERT Model...")
start = time.time()
s_bert = load_sbert(CFG.SBERT_MODEL_PATH, CFG.SBERT_MODEL_FOLDER)

timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Done! ({time.time()-start:.3f}s)")

# Prepare dataset
timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Preparing Dataset...")
start = time.time()
df = get_data(CFG.DATA_PATH, CFG.MODEL_CSV_FILE)
X_train, y_train, X_test, y_test = data_split(dataframe=df,
                                              test_size=0.1,
                                              random_state=42)

# Convert to vector
print("[INFO] Vectorize Data...")
X_train_vectorized = s_bert.encode(X_train[:, 1].tolist(), device=device)
X_test_vectorized = s_bert.encode(X_test[:, 1].tolist(), device=device)

# Create Dataset and DataLoader
train_dataset = ClassifierDataset(X_train_vectorized, y_train)
test_dataset = ClassifierDataset(X_test_vectorized, y_test)

train_dataloader = DataLoader(train_dataset,
                              batch_size=CFG.BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(test_dataset,
                             batch_size=CFG.BATCH_SIZE,
                             shuffle=False)

timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Done! ({time.time()-start:.3f}s)")

# model define
classifier = SimpleMLC(n_classes=CFG.N_CLASSES).to(device)
optimizer = torch.optim.Adam(params=classifier.parameters(), lr=5e-4)
loss_fn = torch.nn.BCEWithLogitsLoss(
    weight=(torch.FloatTensor(
        len(y_train) - y_train.sum(axis=0)) / y_train.sum(axis=0)
    ).to(device)
)

now = datetime.datetime.now().strftime("%y-%m-%d_%H%M")
save_path = f"{CFG.CLASSIFIER_MODEL_PATH}/{now}"
model_name = f"{CFG.CLASSIFIER_MODEL_FILE}.pth"

# Training
timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Training...")

train(classifier,
      train_dataloader=train_dataloader,
      test_dataloader=test_dataloader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      writer=SummaryWriter(log_dir=f"{save_path}/runs/"),
      device=device,
      epochs=CFG.EPOCHS,
      patience=CFG.PATIENCE)

classifier.load_state_dict(classifier.best_params)

save_model(classifier, save_path, model_name)

timestamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
print(f"[INFO] ({timestamp}) Done!")
print(f"[INFO] ({timestamp}) Model saved '{save_path}/{model_name}'")

torch.cuda.empty_cache()