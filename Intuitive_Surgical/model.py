import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class IntSurClassifier(nn.Module):
    def __init__(self,
                num_cls = 4,
                p = 0.1,
                ):

        super().__init__()

        self.num_cls = num_cls
        self.p = p

        self.conv2D1 = nn.Conv2d(3, 8, (3, 3), stride=(2,2), padding=0)  # 3,720,1280 -> 8,639,359
        self.pool2D1 = nn.MaxPool2d(2, 2)                                     # 8,639,359  -> 8,319,179
        self.conv2D2 = nn.Conv2d(8, 16, (3, 3), stride=(2,2), padding=0) # 8,319,179   -> 16,159,89 
        self.pool2D2 = nn.MaxPool2d(2, 2)                                     # 16,159,89   -> 16,79,44
        self.conv2D3 = nn.Conv2d(16, 16, (3, 3), stride=(2,2), padding=0)# 16,79,44    -> 16,39,21
        self.pool2D3 = nn.MaxPool2d(2, 2)                                     # 16,39,21    -> 16,19,10
        self.conv2D4 = nn.Conv2d(16, 32, (3, 3), stride=(2,2), padding=0)# 16,19,10    -> 32,9,4

        self.dropout = nn.Dropout(self.p)

        fc_dim = 32 * 9 * 4
        self.fc1 = nn.Linear(fc_dim, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, self.num_cls)

    def forward(self, x):

        x = F.relu(self.conv2D1(x))
        x = self.pool2D1(x)
        x = F.relu(self.conv2D2(x))
        x = self.pool2D2(x)
        x = self.dropout(x)
        x = F.relu(self.conv2D3(x))
        x = self.pool2D3(x)
        x = F.relu(self.conv2D4(x))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        output = self.fc4(x)

        return output
