import numpy as np
import glob
import os
from random import sample
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


NUM_WORKER = os.cpu_count() // 4

class IntSurDataset(Dataset):

    def __init__(self, fileList = None):

        self.fileList = fileList

    def __len__(self):
        
        return len(self.fileList)

    def __getitem__(self, idx):

        item = np.load(self.fileList[idx]) # each item has two keywords, "img" and "label"
        x = item["img"]                   # H W C = 1280, 720, 3
        y = item["label"]
        x = torch.from_numpy(x)                 
        x = x.permute(2, 0, 1)             # C H W = 3, 1280, 720
        
        return dict(
                    x = x.float(),
                    y = torch.tensor(y).long()
                    )


class IntSurDataModule(pl.LightningDataModule):
    
    """
    IntSurDataModule to setup the train, val, and test data loader  
    
    Parameters
    ----------
    frames_path : str
        the path to the train or test frames
    
    batch_size : int
    
    shuffle : bool
        if True, shuffle the train data only
            
    """
    
    def __init__(self, frames_path = "./Release_v1/frames/",\
                                       batch_size = 4, shuffle = True):
        super().__init__()
        
        self.frames_path = frames_path
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage = None):
        
        
        frameList = glob.glob(self.frames_path + "/*.npz")
        np.random.shuffle(frameList)
        
        
        if stage == "fit":
            trainList, valList, testList = np.split(frameList, 
                                     [int(.8*len(frameList)), int(.9*len(frameList))])
            
            self.train_dataset = IntSurDataset(trainList)
            self.val_dataset   = IntSurDataset(valList)
            self.test_dataset  = IntSurDataset(testList)
            
        if stage == "predict":
            
            frameList.sort()
            self.test_dataset  = IntSurDataset(frameList)
    

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size,\
                            shuffle = True,\
                            num_workers = NUM_WORKER,\
                            drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size,\
                            shuffle = False,\
                            num_workers = NUM_WORKER,\
                            drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size,\
                            shuffle = False,\
                            num_workers = NUM_WORKER,\
                            drop_last=True)