import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint,\
                                        EarlyStopping,\
                                        LearningRateMonitor,\
                                        Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


pl.seed_everything(2)


NUM_WORKER = os.cpu_count() // 2
NUM_Cls = 8
BATCH_SIZE = 8
LR = 1e-6


# Initializing the tokenizer from pretrained BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Reading the data
datapath = 'customer_chat_sample.csv'
df = pd.read_csv(datapath)

# labels and their frequencies to check the data imbalance
uniq_labels = df.label.unique()


# labels dictionary to map each class string to an integer and vice versa
labels2idx = {uniq_labels[i]: i for i in range(len(uniq_labels))}
idx2labels = {v: k for k, v in labels2idx.items()}


# Dividing the data into train, val, and test set, by 80, 10, 10 ratio
df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), 
                                     [int(.8*len(df)), int(.9*len(df))])


# weights list to map each class index to the class weight for weighted loss function                       
weights = [1.0/(df_train["label"]==uniq_labels[i]).sum() \
                        for i in range(len(uniq_labels))]


# If interested to see the barchart of the class frequency
# fig = df_train.groupby(['label']).size().plot(kind='bar',  
#         figsize=(20, 16), fontsize=26).get_figure()

# fig.savefig('class_frequency_barchart.pdf')


# Creating the ChatDataset to grab one data point, and return its tokenzied list and label

class ChatDataset(Dataset):

    def __init__(self, df):

        self.labels = [labels2idx[label] for label in df['label']]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = self.texts[idx]
        y = np.array(self.labels[idx])

        return x, y

    
class CarvanaDataModule(pl.LightningDataModule):
    
    """
    CarvanaDataModule to setup the train, val, and test data loader  
    
    Parameters
    ----------
    df_train : pd.DataFrame
    
    df_val : pd.DataFrame
    
    df_test : pd.DataFrame
    
    batch_size : int
    
    shuffle : bool
        if True, shuffle the train data only
            
    """
    
    def __init__(self, df_train, df_val , df_test, batch_size = 8, shuffle = True):
        super().__init__()
        
        

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage = None):
        
        if stage == "fit" or stage is None:
            self.train_dataset = ChatDataset(self.df_train)
            self.val_dataset   = ChatDataset(self.df_val)
        if stage == "test" or stage is None:
            self.test_dataset  = ChatDataset(self.df_test)
            
        if stage == "predict" or stage is None:
            self.val_dataset   = ChatDataset(self.df_val)
            self.test_dataset  = ChatDataset(self.df_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size,
                            shuffle = True,
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
    
# Creating the LightningBertClassifier with pretarined Bert model as the backbone
    
class LightningBertClassifier(pl.LightningModule):
    
    """
    Bert based classifier
    
    Parameters
    ----------
    LR : float
        learning rate
        
    num_cls : int
        number of the labels or classes
        
    wd: float
        weight decay if AdamW is used
    
    bertModel : str
        BERT model backbone string to be called by transformer
        
    cls_weight : List
        list containing the different class weights for the loss function
    """

    def __init__(self, num_cls = NUM_Cls, dropout = 0.1,\
                 LR = 1e-6,\
                 wd = 0.01,\
                 bertModel = 'bert-base-cased',\
                 cls_weight = [1.0/NUM_Cls] * NUM_Cls):
        super().__init__()

        self.num_cls = num_cls
        self.dropout = dropout
        self.LR = LR
        self.wd = wd
        self.BertPreTrained = bertModel
        self.cls_weight = torch.Tensor(cls_weight)

        self.bert = BertModel.from_pretrained(self.BertPreTrained)
        self.dropout = nn.Dropout(self.dropout)
        self.linear = nn.Linear(768, self.num_cls)
        



    def forward(self, x):

        mask = x['attention_mask']
        input_id = x['input_ids'].squeeze(1)

        _, pooled_output = self.bert(input_ids= input_id, attention_mask = mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)

        return linear_output

    def predict(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        return self(x)

    def cross_entropy_loss(self, logits, labels):
        criterion = nn.CrossEntropyLoss(
                                        weight = self.cls_weight.to(self.device)
                                        )
        return criterion(logits, labels)

    def training_step(self, train_batch, batch_idx):

        x, y = train_batch
        logits = self.forward(x)
        
        loss = self.cross_entropy_loss(logits, y)
        
        predictions = torch.argmax(logits, dim = 1)
        step_accuracy = accuracy(predictions, y)
        
        self.log("train_loss", loss, on_epoch=True, prog_bar = True)
        self.log("train_accuracy", step_accuracy, on_epoch=True, prog_bar = True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, val_batch, batch_idx):

        x, y = val_batch
        logits = self.forward(x)
        
        loss = self.cross_entropy_loss(logits, y)
        
        predictions = torch.argmax(logits, dim = 1)
        step_accuracy = accuracy(predictions, y)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar = True)
        self.log("val_accuracy", step_accuracy, on_epoch=True, prog_bar = True)
        return {"loss": loss, "accuracy": step_accuracy}

    def test_step(self, test_batch, batch_idx):
        
        x, y = test_batch
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



# Train function
def train(lr, batch_size):
    
    print(f"LR = {lr}, BSIZE = {batch_size}")
    checkpoint_path = f"./pl_models/carvana/checkpoint_ADAM_lr={lr}_bSize={batch_size}"
    checkpoint_callback = ModelCheckpoint(
        dirpath = checkpoint_path,
        # filename = "best_checkpoint_",
        save_top_k = 1,
        save_weights_only = True,
        verbose = True,
        monitor = "val_loss",
        mode = "min"
    )
    
    # learning_rate monitor to reduce the lr if the val_loss hasn't been improved
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Tensorboard logger
    logger = TensorBoardLogger("lightning_logs", name = f"Carvana_lr={lr}_bSize={batch_size}")
    
    # Stop training if the val_loss is increasing for 4 consecutive epochs
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=4)


    data_module = CarvanaDataModule(df_train, df_val, df_test, batch_size = batch_size, shuffle = True)

    data_module.setup()

    # train
    model = LightningBertClassifier(LR = lr, cls_weight = weights)
    trainer = Trainer(
        logger = logger,
        max_epochs = 30 ,
        callbacks= [checkpoint_callback, early_stopping_callback, lr_monitor],
        gpus=1,
        accelerator=None,
        progress_bar_refresh_rate = 30)

    trainer.fit(model, data_module)
    
    # Model evaluation metrics

    val_module, test_module = data_module.val_dataloader(), data_module.test_dataloader()
    
    # eval to turn off the dropout for prediction step
    model.eval()
    
    pred_val = trainer.predict(model, dataloaders = val_module) # 
    pred_test = trainer.predict(model, dataloaders = test_module) # 
    
    Y_val = []
    Y_test = []

    for item in tqdm(val_module):
        Y_val.append(item[1].numpy())

    for item in tqdm(test_module):
        Y_test.append(item[1].numpy())
        
    y_val = Y_val[0]
    for elm in tqdm(Y_val[1:]):
        y_val = np.hstack((y_val, elm))

    y_test = Y_test[0]
    for elm in tqdm(Y_test[1:]):
        y_test = np.hstack((y_test, elm))
        
    P_val = []
    P_test = []

    for item in tqdm(pred_val):
        P_val.append(item)

    for item in tqdm(pred_test):
        P_test.append(item)
        
        
    p_val = np.argmax(P_val[0], axis = 1)
    for elm in tqdm(P_val[1:]):

        p_val = np.hstack((p_val, np.argmax(elm, axis = 1)))

    p_test = np.argmax(P_test[0], axis = 1)
    for elm in tqdm(P_test[1:]):
        p_test = np.hstack((p_test, np.argmax(elm, axis = 1)))
        
    val_accuracy = accuracy(torch.from_numpy(p_val), torch.from_numpy(y_val))
    test_accuracy = accuracy(torch.from_numpy(p_test), torch.from_numpy(y_test))
    
    
    acc_file = f"./pl_models/carvana/accuracy_ADAM_lr={lr}_bSize={batch_size}.txt"
    with open(acc_file, 'w') as f:
        f.write("val_accuracy: " + str(val_accuracy) + '\n')
        f.write("test_accuracy: " + str(test_accuracy))
    
    confusion_matrix_val = pd.DataFrame(confusion_matrix(y_val, p_val))\
    .rename(columns=idx2labels, index=idx2labels)
    
    filepathval = f"./pl_models/carvana/confusion_matrix_val_ADAM_lr={lr}_bSize={batch_size}.csv"
    
    confusion_matrix_val.to_csv(filepathval, index=False)
    
    confusion_matrix_test = pd.DataFrame(confusion_matrix(y_test, p_test))\
    .rename(columns=idx2labels, index=idx2labels)
    
    filepathtest = f"./pl_models/carvana/confusion_matrix_test_ADAM_lr={lr}_bSize={batch_size}.csv"
    confusion_matrix_test.to_csv(filepathtest, index=False)
    
    
    test_report = classification_report(y_test, p_test, output_dict=True)
    val_report = classification_report(y_val, p_val, output_dict=True)
    
    
    dfVal = pd.DataFrame(val_report).transpose()
    dfTest = pd.DataFrame(test_report).transpose()
    
    path_report_val = f"./pl_models/carvana/report_val_ADAM_lr={lr}_bSize={batch_size}.csv"
    path_report_test = f"./pl_models/carvana/report_test_ADAM_lr={lr}_bSize={batch_size}.csv"
    
    dfVal.to_csv(path_report_val, index=False)
    dfTest.to_csv(path_report_test, index=False)



if __name__ == "__main__":

    lrList = [
        1e-6,
        ]
    batch_sizeList = [
                    8, 
                    ]



    for lr in lrList:
        for batch_size in batch_sizeList:
            train(lr, batch_size)