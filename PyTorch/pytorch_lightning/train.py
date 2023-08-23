import os

import pandas as pd
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models import EfficientDetModel
from dataset import (
    CarsDatasetAdaptor,
    CarsDatasetAdaptorInference
)
from datamodules import EfficientDetDataModule
from params import *


def main():
    df = pd.read_csv(os.path.join(DATA_PATH, TRAIN_CSV))

    model = EfficientDetModel(model_architecture=ARCHITECTURE)

    train_ds = CarsDatasetAdaptor(TRAIN_PATH, df)
    valid_ds = CarsDatasetAdaptor(TRAIN_PATH, df)
    predict_ds = CarsDatasetAdaptorInference(TEST_PATH)

    dm = EfficientDetDataModule(
        train_dataset_adaptor=train_ds,
        validation_dataset_adaptor=valid_ds,
        predict_dataset_adaptor=predict_ds
    )

    logger = TensorBoardLogger("../tb_logs", name="cars_detect")
    profiler = PyTorchProfiler(on_trace_ready=torch.profiler.tensorboard_trace_handler("../tb_logs/profiler0"),
                               schedule=torch.profiler.schedule(skip_first=1, wait=1, warmup=1, active=3))
    callbacks = [
        ModelCheckpoint(
            monitor="valid_loss",
            filename="cars-detector_{epoch:02d}-{valid_loss:.4f}",
            save_last=False,
            save_top_k=3,
            mode="min"
        ),
        EarlyStopping(
            monitor="valid_loss",
            patience=5,
            mode="min",
        )
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        precision="16-mixed",
        max_epochs=20,
        # profiler=profiler,
        logger=logger,
        callbacks=callbacks,
        )

    trainer.fit(model, dm)
    trainer.validate(model=model, datamodule=dm);


if __name__ == "__main__":
    main()