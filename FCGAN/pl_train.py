import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from model import Discriminator, Generator

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10
        
    def prepare_data(self):
        # downlaod
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)
        
    def setup(self, stage = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
            
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size = self.batch_size, num_workers = self.num_workers,)
    
    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size = self.batch_size, num_workers = self.num_workers,)
    
    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size = self.batch_size, num_workers = self.num_workers,)


class FCGAN(pl.LightningModule):
    def __init__(self,
                 channels,
                 width,
                 height,
                 z_dim: int = 64,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = BATCH_SIZE,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        # networks
        data_shape = (channels, width, height)
        self.gen = Generator(z_dim = z_dim, img_dim = np.prod(data_shape))
        self.disc = Discriminator(in_features = np.prod(data_shape))
        
        # N x channels_noise 
        self.validation_z = torch.randn(8, self.hparams.z_dim)
        self.example_input_array = torch.zeros(2, self.hparams.z_dim)
        
    def forward(self, z):
        return self.gen(z)
        
    def adv_loss(self, y_hat, y):

        criterion = nn.BCELoss()
        return criterion(y_hat, y)
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch
        real = real.view(-1, self.hparams.width * self.hparams.height)

        # sample noise
        z = torch.randn(real.shape[0], self.hparams.z_dim)
        z = z.type_as(real)
        fake = self(z)
        
        # train discriminator
        if optimizer_idx == 0:
            # Measure discriminator's ability to classify real from generated samples
            # how well can it label as real?
            
            disc_real = self.disc(real).view(-1)
            loss_disc_real = self.adv_loss(disc_real, torch.ones_like(disc_real).type_as(disc_real))
            disc_fake = self.disc(fake.detach()).view(-1)
            loss_disc_fake = self.adv_loss(disc_fake, torch.zeros_like(disc_fake).type_as(disc_fake))
            
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
       
            tqdm_dict = {"d_loss": loss_disc}
            output = OrderedDict({"loss": loss_disc, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train generator
        if optimizer_idx == 1:

            # adversarial loss is binary cross-entropy
            output = self.disc(fake).view(-1)
            
            g_loss = self.adv_loss(output, torch.ones_like(output).type_as(output))
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        opt_g = torch.optim.Adam(self.gen.parameters(), lr = lr, betas = (b1,b2))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(b1, b2))
        
        return [opt_d, opt_g], []
    
    def on_epoch_end(self):
        z = self.validation_z.type_as(self.gen.gen[0].weight)
        
        # log samples images
        sample_imgs = self(z)
        torch.save(sample_imgs, 'sample_imgs.pt')
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch) 



if __name__ == '__main__':
	dm = MNISTDataModule()
	model = FCGAN(*dm.size())
	trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=50, progress_bar_refresh_rate=20)
	trainer.fit(model, dm)