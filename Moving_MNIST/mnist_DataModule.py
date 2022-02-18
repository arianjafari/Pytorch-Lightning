import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from MovingMNIST import MovingMNIST
import pytorch_lightning as pl

NUM_WORKER = os.cpu_count() // 2

class MovingMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_root = './moving_MNIST',
                        seq_len = 20, image_size = 64, num_digits = 2,
                        batch_size = 32, shuffle = True):
        super().__init__()

        """
        
        """

        self.data_root = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.num_digits = num_digits
        self.batch_size = batch_size
        self.shuffle = shuffle


    def setup(self):
        
        self.train_dataset = MovingMNIST(train=True,
                                          data_root=self.data_root,
                                          seq_len=self.seq_len,
                                          image_size=self.image_size,
                                          deterministic=True,
                                          num_digits=self.num_digits)
        
        # both val and test sets are the same 
        self.val_dataset = MovingMNIST(train=False,
                                          data_root=self.data_root,
                                          seq_len=self.seq_len,
                                          image_size=self.image_size,
                                          deterministic=True,
                                          num_digits=self.num_digits)
        self.test_dataset = MovingMNIST(train=False,
                                          data_root=self.data_root,
                                          seq_len=self.seq_len,
                                          image_size=self.image_size,
                                          deterministic=True,
                                          num_digits=self.num_digits)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                            shuffle = self.shuffle,
                            num_workers = NUM_WORKER,
                            drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size,
                            shuffle = False,
                            num_workers = NUM_WORKER,
                            drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,
                            shuffle = False,
                            num_workers = NUM_WORKER,
                            drop_last=True)