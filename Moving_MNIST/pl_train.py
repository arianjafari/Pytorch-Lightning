import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from multiprocessing import Process
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint,\
                                        EarlyStopping,\
                                        LearningRateMonitor,\
                                        Callback
from pytorch_lightning.loggers import TensorBoardLogger

from mnist_DataModule import MovingMNISTDataModule
from seq_2_seq_convlstm import EncoderDecoderConvLSTM
pl.seed_everything(2)


#----------create some dirs---------
teacher_forcing = False
LSTM_DIMS = [64, 64, 64]
root_log_dir = './Lightning_res_' + ('' if teacher_forcing else 'ntf_') + \
                                                    str(LSTM_DIMS)[1:-1].replace(', ','x')

exists = True
while exists:
    exists = os.path.exists(root_log_dir)
    if not exists:
        os.makedirs(root_log_dir)
    else:
        root_log_dir += '_'
print(os.path.exists(root_log_dir))
train_out_dir = root_log_dir + '/train'
val_out_dir = root_log_dir + '/val'

chkpnt_dir = root_log_dir + '/checkpoint'

os.makedirs(train_out_dir)
os.makedirs(val_out_dir)
os.makedirs(chkpnt_dir)

log_file = root_log_dir + '/out.log'
f = open(log_file,"w+")
f.close()




MAX_EPOCH = 100


HIDDEN_DIM = 64
BATCH_SIZE = 64
HIDDEN_SPT = 16
LSTM_DIMS = [64, 64, 64]
IMG_SIZE = 64
IMG_CHANNEL = 1
OUT_CHANNEL = 1
PRINT_STEP = 500

SEQ_LEN = 20


LEARNING_RATE = 1e-4
BETA_1 = 0.9
BETA_2 = 0.98

class MovingMNISTLightning(pl.LightningModule):

    def __init__(self,
                 hidden_dim = HIDDEN_DIM,
                 batch_size = BATCH_SIZE,
                 hidden_spt = HIDDEN_SPT,
                 lstm_dims = LSTM_DIMS,
                 img_size = IMG_SIZE,
                 img_channel = IMG_CHANNEL,
                 out_channel = OUT_CHANNEL,
                 print_step = PRINT_STEP,
                 lr = LEARNING_RATE,
                 teacher_forcing = True,
                 log_images = True,
                 beta_1 = BETA_1,
                 beta_2 = BETA_2,
                ):
        
        super(MovingMNISTLightning, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden_spt = hidden_spt
        self.lstm_dims = lstm_dims
        self.img_size = img_size
        self.img_channel = img_channel
        self.out_channel = out_channel
        self.print_step = print_step
        self.teacher_forcing = teacher_forcing
        
        self.n_steps_past = 10
        self.n_steps_ahead = 10
        
        
        self.log_images = log_images
        self.save_hyperparameters()
        
        self.model = EncoderDecoderConvLSTM(hidden_dim = self.hidden_dim,
                                            batch_size = self.batch_size,
                                            hidden_spt = self.hidden_spt,
                                            lstm_dims = self.lstm_dims,
                                            img_size = self.img_size,
                                            img_channel = self.img_channel,
                                            out_channel = self.out_channel,
                                            print_step = self.print_step,
                                            teacher_forcing = self.teacher_forcing,)

        

    def forward(self, x, step = 0, is_train = True):

        output = self.model(x, future_seq = self.n_steps_ahead, step = step, is_train = is_train)

        return output
    
    def loss_fn(self, logits, labels):
        criterion = nn.BCELoss()
        return criterion(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch, batch[:, self.n_steps_past:, :, :, :]

        y_hat = self.forward(x, step = self.global_step, is_train = True)
        
        loss = self.loss_fn(y_hat, y)

#         # save learning_rate
        lr_saved = self.trainer.optimizers[0].param_groups[-1]['lr']
        lr_saved = torch.scalar_tensor(lr_saved).cuda()

        # save predicted images every 250 global_step
        if self.log_images:
            if self.global_step % 250 == 0:
                l = []
                for j in range(3):
                    l.append(x[j, : self.n_steps_past, :, :, :].cpu())
                    l.append(y_hat[j,...].data.cpu())
                    l.append(y[j,...].cpu())
                samples = torch.cat(l).cpu()
                torchvision.utils.save_image(samples,
                                         train_out_dir + "/{0:0>5}iter.png".format(self.global_step), nrow=10)

        self.log("train_loss", loss, on_epoch=True, prog_bar = True)
        return {'loss': loss, 'learning_rate': lr_saved}
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch, val_batch[:, self.n_steps_past:, :, :, :]
        
        y_hat = self.forward(x, step = self.global_step, is_train = False)

        loss = self.loss_fn(y_hat, y)
        
        # save predicted images every 250 global_step
        if self.log_images:
            if batch_idx == 0:
                l = []
                for j in range(3):
                    l.append(x[j, : self.n_steps_past, :, :, :].cpu())
                    l.append(y_hat[j,...].data.cpu())
                    l.append(y[j,...].cpu())
                samples = torch.cat(l).cpu()
                torchvision.utils.save_image(samples,
                                         val_out_dir + "/{0:0>5}iter.png".format(self.global_step), nrow=10)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar = True)
        return {'val_loss': loss,}
    
#     def test_step(self, batch, batch_idx):
#         # OPTIONAL
#         x, y = batch
#         y_hat = self.forward(x)
#         return {'test_loss': self.criterion(y_hat, y)}


#     def test_end(self, outputs):
#         # OPTIONAL
#         avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
#         tensorboard_logs = {'test_loss': avg_loss}
#         return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr = self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2))
        lr_scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,\
                                                                patience = 8, verbose = True),
        'monitor': 'val_loss'}
        self.log("learning_rate", optimizer.state_dict()["param_groups"][0]['lr'],\
                     on_epoch=True, prog_bar = True
                )
        
#         return optimizer
        return [optimizer], [lr_scheduler]



def run_trainer():
    
    data_module = MovingMNISTDataModule(data_root = './moving_MNIST',
                        seq_len = SEQ_LEN, image_size = IMG_SIZE, num_digits = 2,
                        batch_size = BATCH_SIZE, shuffle = True)
    data_module.setup()
    train_module = data_module.train_dataloader()
    val_module = data_module.val_dataloader()
    test_module = data_module.test_dataloader()

    
    
    checkpoint_path = "./mnist_models_pl/checkpoint"
    checkpoint_callback = ModelCheckpoint(
    dirpath = checkpoint_path,
    # filename = "best_checkpoint_",
    save_top_k = 1,
    save_weights_only = True,
    verbose = True,
    monitor = "val_loss",
    mode = "min")
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger("lightning_logs", name = "MNIST")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=6)

    model = MovingMNISTLightning()

    trainer = Trainer(
                      logger = logger,
                      max_epochs = MAX_EPOCH ,
                      callbacks= [checkpoint_callback, lr_monitor,],
                      gpus=1,
                      accelerator=None,
                      progress_bar_refresh_rate = 30)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    run_trainer()
