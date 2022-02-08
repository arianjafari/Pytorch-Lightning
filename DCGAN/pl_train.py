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

from model import Discriminator, Generator, initialize_weights

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

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
                transforms.Normalize((0.1307,), (0.3081,)),
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



class DCGAN(pl.LightningModule):
    def __init__(self,
                 channels,
                 width,
                 height,
                 features_g = 64,
                 features_d = 64,
                 channels_noise: int = 100,
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
        self.generator = Generator(channels_noise = channels_noise,\
                                   channels_img = channels, features_g = features_g)
        self.discriminator = Discriminator(channels_img = channels, features_d = features_d)
        
        # N x channels_noise x 1 x 1
        self.validation_z = torch.randn(8, self.hparams.channels_noise, 1, 1)
        self.example_input_array = torch.zeros(2, self.hparams.channels_noise, 1, 1)
        
    def forward(self, z):
        return self.generator(z)
        
    def adversarial_loss(self, y_hat, y):

        criterion = nn.BCELoss()
        return criterion(y_hat, y)
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.channels_noise, 1, 1)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log samples images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop

            # adversarial loss is binary cross-entropy
            output = self.discriminator(self(z)).reshape(-1)
            valid = torch.ones_like(output)
            valid = valid.type_as(output)
            
            g_loss = self.adversarial_loss(output, valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            # how well can it label as real?
            
            disc_real = self.discriminator(imgs).reshape(-1)
            valid = torch.ones_like(disc_real)
            valid = valid.type_as(disc_real)

            real_loss = self.adversarial_loss(disc_real, valid)

            # how well can it label as fake?
            
            disc_fake = self.discriminator(self(z).detach()).reshape(-1)
            fake = torch.zeros_like(disc_fake)
            fake = fake.type_as(disc_fake)

            fake_loss = self.adversarial_loss(disc_fake, fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        opt_g = torch.optim.Adam(self.generator.parameters(), lr = lr, betas = (b1,b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        
        return [opt_g, opt_d], []
    
    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.gen[0][0].weight)
        
        # log samples images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)



if __name__ == '__main__':
	
	dm = MNISTDataModule()
	model = DCGAN(*dm.size())
	trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=50, progress_bar_refresh_rate=20)
	trainer.fit(model, dm)
