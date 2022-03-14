import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pytorch_lightning as pl
import glob
from tqdm import tqdm
import math

NUM_WORKER = os.cpu_count() // 2

class AriaDataset(Dataset):
    def __init__(self, sequences, features_idx = [0,1,2,3,4]):
        self.sequences = sequences
        self.features_idx = features_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        x_seq = []
        y_seq = []
        
        # to loop over input sequence and stack them as a torch tensor with the shape of T,C,H,W
        for file in sequence:
            item = np.load(file)
            x_seq.append(torch.tensor(item["X"][self.features_idx,...]))
        
        # to loop over label sequence and stack them as a torch tensor with the shape of T,H,W
        for file in label:
            item = np.load(file)
            y_seq.append(torch.tensor(item["y"]))
        
        x_seq = torch.stack(x_seq)
        y_seq = torch.stack(y_seq)
        
        return  dict(X = x_seq, y = y_seq.unsqueeze(1)) # to make y from T,H,W to T,1,H,W

class AriaStDataModule(pl.LightningDataModule):
    def __init__(self, data_rootList = ["../data/features_labels/2018_2019_PCT/"],
                        features_idx = [0,1,2,3,4],
                        seq_len = 10, label_len = 10, train_cut = 0.8,
                        batch_size = 32, shuffle = True):
        super().__init__()

        """
        data_rootList: contains a list of the all paths to the input data. Assume that we have data for 
        year=2018, month = 1,2,3,4. As there might be a gap between the sequence data from month i to month i+1,
        we can not put them all in one folder because it messes the consistency of the sequences. Therefore, treating each
        month separately, we can build consistent sequences for each month and then concatenate them as an input to 
        the model
        
        """

        self.data_rootList = data_rootList
        self.features_idx = features_idx
        self.seq_len = seq_len
        self.label_len = label_len
        self.train_cut = train_cut
        self.batch_size = batch_size
        self.shuffle = shuffle
        


    def create_sequences(self, input_data: list, seq_len: int = 10, label_len: int = 10):
        sequences = []
        data_size = len(input_data)

        for i in range(data_size - (seq_len + label_len)):
            sequence = input_data[i:i+seq_len]

            label_position = i + seq_len
            label = input_data[label_position:label_position+label_len]

            sequences.append((sequence, label))

        return sequences
    
    
    def setup(self):
        all_sequences = []
        
        for data_root in self.data_rootList:
            file_list = sorted(glob.glob(data_root + "*.npz"))
        
            all_sequences += self.create_sequences(input_data = file_list,\
                                              seq_len = self.seq_len,\
                                              label_len = self.label_len)
        
        self.train_sequences = all_sequences[: math.floor(self.train_cut*len(all_sequences))]
        self.test_sequences = all_sequences[math.floor(self.train_cut*len(all_sequences)) :]
        
        
        
        # both val and test sets are the same
        self.train_dataset = AriaDataset(self.train_sequences, self.features_idx)
        self.val_dataset = AriaDataset(self.test_sequences, self.features_idx)
        self.test_dataset = AriaDataset(self.test_sequences, self.features_idx)


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
