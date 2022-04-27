import numpy as np
import glob
import os
from random import sample
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from tqdm import tqdm
import skvideo.io
from pytorch_lightning.callbacks import ModelCheckpoint,\
                                        EarlyStopping,\
                                        LearningRateMonitor,\
                                        Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from dataModule import IntSurDataModule
from model import IntSurClassifier

import sys
import argparse
import mlflow

pl.seed_everything(2) # setting random seed
# NUM_WORKER = os.cpu_count() // 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# if cuda is available, get one gpu for training
if torch.cuda.is_available():
    GPU = 1
else:
    GPU = 0

parser = argparse.ArgumentParser()

parser.add_argument("--checkpoint_path", default = "./pl_models/Intuitive/checkpoint/",\
                    help="checkpoint dir to store the best model")
parser.add_argument("--epoch", default = 5, help="number of epoch")
parser.add_argument("--train_path", default = "./Release_v1/frames/", help="trainset path")
parser.add_argument("--test_path", default = "./Release_Test/frames/", help="testset path")
parser.add_argument("--batch_size", default = 1, help="batch_size")

args = parser.parse_args(sys.argv[1:])

train_path = args.train_path
test_path = args.test_path
batch_size = int(args.batch_size)
MAX_EPOCH = int(args.epoch)
NUM_Cls = 4



class LightningIntSurClassifier(pl.LightningModule):
    """
    A simple classifier to just test the ML pipeline,
    The problme is an object detection type of problem,
    but here we just assumed a simple classification problem.
    
    Parameters
    ----------
    LR : float
        learning rate
        
    num_cls : int
        number of the labels or classes

    p : float
        dropout probability
        
    wd: float
        weight decay if AdamW is used
        
    cls_weight : List
        list containing the different class weights for the loss function
    """

    def __init__(self, num_cls = NUM_Cls,\
                 p = 0.1,\
                 LR = 1e-6,\
                 wd = 0.01,\
                 cls_weight = [1.0/NUM_Cls] * NUM_Cls):

        super().__init__()

        self.num_cls = num_cls
        self.p = p
        self.LR = LR
        self.wd = wd
        self.cls_weight = torch.Tensor(cls_weight)
        
        self.model = IntSurClassifier(num_cls = 4,
                                      p = 0.1,)
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def predict(self, batch, batch_idx, dataloader_idx):
        x, y = batch["x"], batch["y"]
        return self(x)

    def cross_entropy_loss(self, logits, labels):
        criterion = nn.CrossEntropyLoss(
                                        weight = self.cls_weight.to(self.device)
                                        )
        return criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):

        x, y = train_batch["x"], train_batch["y"]
        logits = self.forward(x)
        
        loss = self.cross_entropy_loss(logits, y)
        
        predictions = torch.argmax(logits, dim = 1)
        step_accuracy = accuracy(predictions, y)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar = True)
        self.log("train_accuracy", step_accuracy, on_epoch=True, prog_bar = True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, val_batch, batch_idx):

        x, y = val_batch["x"], val_batch["y"]
        logits = self.forward(x)
        
        loss = self.cross_entropy_loss(logits, y)
        
        predictions = torch.argmax(logits, dim = 1)
        step_accuracy = accuracy(predictions, y)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar = True)
        self.log("val_accuracy", step_accuracy, on_epoch=True, prog_bar = True)
        return {"loss": loss, "accuracy": step_accuracy}

    def test_step(self, test_batch, batch_idx):
        
        x, y = test_batch["x"], test_batch["y"]
        logits = self.forward(x)
        
        loss = self.cross_entropy_loss(logits, y)
        
        predictions = torch.argmax(logits, dim = 1)
        step_accuracy = accuracy(predictions, y)
        
        self.log("test_loss", loss, on_epoch=True, prog_bar = True)
        self.log("test_accuracy", step_accuracy, on_epoch=True, prog_bar = True)
        return {"loss": loss, "accuracy": step_accuracy}

    def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.parameters(), lr=self.LR, weight_decay=self.wd)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
        lr_scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = 2, verbose = True),
        'monitor': 'val_loss'}
        return [optimizer], [lr_scheduler]


checkpoint_path = args.checkpoint_path

checkpoint_callback = ModelCheckpoint(
    dirpath = checkpoint_path,
    save_top_k = 1,
    save_weights_only = True,
    verbose = True,
    monitor = "val_loss",
    mode = "min"
) 


# learning_rate monitor to reduce the lr if the val_loss hasn't been improved
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Tensorboard logger
logger = TensorBoardLogger("lightning_logs", name = "Intuitive")

# Stop training if the val_loss is increasing for 4 consecutive epochs
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=4)

data_module = IntSurDataModule(frames_path = train_path, batch_size = batch_size) 
data_module.setup(stage = "fit")
train_module, val_module, test_module = data_module.train_dataloader(),\
data_module.val_dataloader(), data_module.test_dataloader()

# print(len(train_module), len(val_module), len(test_module))

model = LightningIntSurClassifier(LR = 5e-5)
trainer = Trainer(
    logger = logger,
    max_epochs = MAX_EPOCH ,
    callbacks= [checkpoint_callback, early_stopping_callback, lr_monitor],
    gpus=GPU,
    accelerator=None,
    progress_bar_refresh_rate = 30)

trainer.fit(model, data_module)