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


# Hyperparameters etc
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

# device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 5e-5
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 128
NUM_EPOCHS = 5
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
        IMAGE_SIZE: int = IMAGE_SIZE
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.IMAGE_SIZE = IMAGE_SIZE
        
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, self.IMAGE_SIZE, self.IMAGE_SIZE)
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



class WGAN(pl.LightningModule):
    def __init__(self,
                 channels,
                 width,
                 height,
                 features_g = FEATURES_GEN,
                 features_d = FEATURES_CRITIC,
                 channels_noise: int = Z_DIM,
                 lr: float = LEARNING_RATE,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = BATCH_SIZE,
                 clip_value: float = WEIGHT_CLIP,
                 n_critic: int = CRITIC_ITERATIONS,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        # networks
        data_shape = (channels, width, height)
        self.gen = Generator(channels_noise = channels_noise,\
                                   channels_img = channels, features_g = features_g)
        self.critic = Discriminator(channels_img = channels, features_d = features_d)
        
        initialize_weights(self.gen)
        initialize_weights(self.critic)
        
        # N x channels_noise x 1 x 1
        self.validation_z = torch.randn(8, self.hparams.channels_noise, 1, 1)
        self.example_input_array = torch.zeros(2, self.hparams.channels_noise, 1, 1)
        
    def forward(self, z):
        return self.gen(z)
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch

        # sample noise
        z = torch.randn(real.shape[0], self.hparams.channels_noise, 1, 1)
        z = z.type_as(real)
        fake = self(z)
        
        # train discriminator
        if optimizer_idx == 0:
            # Measure discriminator's ability to classify real from generated samples
            # how well can it label as real?
            
            
            critic_real = self.critic(real).reshape(-1)
            critic_fake = self.critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
            
            for p in self.critic.parameters():
                p.data.clamp_(-self.hparams.clip_value, self.hparams.clip_value)

            tqdm_dict = {"critic_loss": loss_critic}
            output = OrderedDict({"loss": loss_critic, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train generator
        if optimizer_idx == 1:

            # adversarial loss is binary cross-entropy
            gen_fake = self.critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            
            tqdm_dict = {"g_loss": loss_gen}
            output = OrderedDict({"loss": loss_gen, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        
    
    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        
        opt_g = torch.optim.Adam(self.gen.parameters(), lr = lr, betas = (b1,b2))
        opt_d = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(b1, b2))
        
        return ({'optimizer': opt_d, 'frequency': self.hparams.n_critic},\
                {'optimizer': opt_g, 'frequency': 1})
    
    def on_before_zero_grad(self, *args, **kwargs):
        pass
    
    def on_epoch_end(self):
        z = self.validation_z.to(self.device)
        
        # log samples images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch) 


if __name__ == "__main__":
	dm = MNISTDataModule()
	model = WGAN(*dm.size())
	trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=50, progress_bar_refresh_rate=20)
	trainer.fit(model, dm)