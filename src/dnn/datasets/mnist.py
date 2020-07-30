import pathlib
import sys

import pytorch_lightning as pl
import torchvision.datasets
import torchvision.transforms


from . import utils

TRAIN_TRANSFORM = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
TRAIN_TRANSFORM = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
TEST_TRANSFORM = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
MNIST_NAME = torchvision.datasets.MNIST.__name__
MNIST_NORMALIZE = True
MNIST_VALIDATION_PERCENTAGE = 0.1

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, path=DEFAULT_DATASET_PATH,
                       val_percentage=MNIST_VALIDATION_PERCENTAGE,
                       transform=None,
                       normalize=MNIST_NORMALIZE):
        self._path = str(pathlib.Path(path).resolve())

    def prepare_data(self):
        # download
        torch.datasets.MNIST(, train=True, download=True, transform=None)
        torch.datasets.MNIST(, train=False, download=True, transform=None)

    def setup(self, stage):
        mnist_train = MNIST(, train=True, download=False, transform=transforms.ToTensor())
        mnist_test = MNIST(, train=False, download=False, transform=transforms.ToTensor())
        # train/val split
        mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=64)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=64)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=64)

DATASET = {MNIST.__name__: MNIST}
