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
# import matplotlib.pyplot as plt
from dataModule import IntSurDataModule
from model import IntSurClassifier

import sys
import argparse
import mlflow

pl.seed_everything(2)
NUM_WORKER = os.cpu_count() // 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)
if torch.cuda.is_available():
    GPU = 1
else:
    GPU = 0

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", default = "./pl_models/Intuitive/checkpoint/",\
                        help="checkpoint dir to store the best model")
parser.add_argument("--test_path", default = "./Release_Test/frames/", help="testset path")
parser.add_argument("--best_model", help="best model ckpt")
parser.add_argument("--batch_size", default = 1, help="batch_size")

args = parser.parse_args(sys.argv[1:])


test_path = args.test_path
best_model = args.best_model
batch_size = int(args.batch_size)
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



model = LightningIntSurClassifier()

checkpoint_path = args.checkpoint_path
saved_path = os.path.join(checkpoint_path, best_model)
# print("saved_path: ", saved_path)
model.load_from_checkpoint(saved_path)
model.eval()
trainer = Trainer(gpus = GPU)

data_module = IntSurDataModule(frames_path = test_path, batch_size = batch_size) 
data_module.setup(stage = "predict")
test_module = data_module.test_dataloader()


predict_test = trainer.predict(model, dataloaders = test_module)

# print(len(predict_test), predict_test[0].shape)

P_test = []

for item in tqdm(predict_test):
    P_test.append(item)

p_test = np.argmax(P_test[0], axis = 1)
for elm in tqdm(P_test[1:]):
    p_test = np.hstack((p_test, np.argmax(elm, axis = 1)))

# the prediction values can be saved in any format