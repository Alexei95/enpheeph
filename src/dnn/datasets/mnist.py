import math
import pathlib
import sys

import pytorch_lightning as pl
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

from . import datamoduleabc


MNIST_TRAIN_TRANSFORM = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
MNIST_VALIDATION_TRANSFORM = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
MNIST_TEST_TRANSFORM = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])
MNIST_TRAIN_VAL_LENGTH = 60000
MNIST_TEST_LENGTH = 10000
MNIST_TRAIN_PERCENTAGE = 0.9
MNIST_VALIDATION_PERCENTAGE = 0.1
MNIST_TEST_PERCENTAGE = 1.0
MNIST_DATASET = torchvision.datasets.MNIST
MNIST_NAME = MNIST_DATASET.__name__
MNIST_SIZE = torch.Size([1, 28, 28])
MNIST_N_CLASSES = len(MNIST_DATASET.classes)


class MNISTDataModule(datamoduleabc.DataModuleABC):
    _name = MNIST_NAME
    _n_classes = MNIST_N_CLASSES
    _size = MNIST_SIZE

    def __init__(self, name=MNIST_NAME,
                       n_classes=MNIST_N_CLASSES,
                       size=MNIST_SIZE,

                       dataset_class=MNIST_DATASET,

                       train_transform=MNIST_TRAIN_TRANSFORM,
                       train_percentage=MNIST_TRAIN_PERCENTAGE,

                       val_transform=MNIST_VALIDATION_TRANSFORM,
                       val_percentage=MNIST_VALIDATION_PERCENTAGE,

                       test_transform=MNIST_TEST_TRANSFORM,
                       test_percentage=MNIST_TEST_PERCENTAGE,

                       *args,
                       **kwargs):
        kwargs.update({'name': name,
                       'dataset_class': dataset_class,
                       'train_transform': train_transform,
                       'train_percentage': train_percentage,
                       'val_transform': val_transform,
                       'val_percentage': val_percentage,
                       'test_transform': test_transform,
                       'test_percentage': test_percentage,})
        super().__init__(*args, **kwargs)

        self._asserts()

        self._train_val_length = MNIST_TRAIN_VAL_LENGTH
        self._test_val_length = MNIST_TEST_LENGTH

        self._train_indices = None
        self._val_indices = None
        # while we don't actually require indices to split the dataset for
        # testing, they are used if a lower testing percentage is used
        self._test_indices = None

    def _asserts(self):
        assert (self._train_percentage + self._val_percentage) > 0
        assert (self._train_percentage + self._val_percentage) <= 1

    def prepare_data(self):
        # download
        self._dataset_class(self._path, train=True, download=True, transform=None)
        self._dataset_class(self._path, train=False, download=True, transform=None)

    def reset_indices(self):
        train_n_indices = math.floor(self._train_percentage * self._train_val_length)
        val_n_indices = math.ceil(self._val_percentage * self._train_val_length)
        train_val_indices = torch.randperm(train_n_indices + val_n_indices).tolist()
        self._train_indices = train_val_indices[0:train_n_indices]
        self._val_indices = train_val_indices[train_n_indices:(train_n_indices + val_n_indices)]

    def setup(self, stage):
        # if we don't have indices setup for training and validation
        # we create new ones
        # indices are not required for testing as we use the whole dataset
        if self._train_indices is None or self._val_indices is None:
            self.reset_indices()
        mnist_train = self._dataset_class(self._path, train=True, download=False, transform=self._train_transform)
        mnist_val = self._dataset_class(self._path, train=True, download=False, transform=self._val_transform)
        mnist_test = self._dataset_class(self._path, train=False, download=False, transform=self._test_transform)

        self._train_dataset = torch.utils.data.Subset(mnist_train, self._train_indices)
        self._val_dataset = torch.utils.data.Subset(mnist_val, self._val_indices)
        self._test_dataset = mnist_test

DATASET = {MNIST_NAME: MNISTDataModule}
