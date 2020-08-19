import pathlib
import sys

import pytorch_lightning as pl
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

from . import datamoduleabc

CIFAR10_TRAIN_TRANSFORM = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.RandomCrop(32, padding=4),
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                [0.2023, 0.1994, 0.2010]),
                        ])
CIFAR10_VALIDATION_TRANSFORM = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010]),
                            ])
CIFAR10_TEST_TRANSFORM = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                [0.2023, 0.1994, 0.2010]),
                        ])
CIFAR10_TRAIN_VAL_LENGTH = 50000
CIFAR10_TEST_LENGTH = 10000
CIFAR10_TRAIN_PERCENTAGE = 0.9
CIFAR10_VALIDATION_PERCENTAGE = 0.1
CIFAR10_TEST_PERCENTAGE = 1.0
CIFAR10_DATASET = torchvision.datasets.CIFAR10
CIFAR10_NAME = CIFAR10_DATASET.__name__
CIFAR10_SIZE = torch.Size([3, 32, 32])
CIFAR10_N_CLASSES = 10


class CIFAR10DataModule(datamoduleabc.DataModuleABC):
    _name = CIFAR10_NAME
    _n_classes = CIFAR10_N_CLASSES
    _size = CIFAR10_SIZE

    def __init__(self, name=CIFAR10_NAME,
                       n_classes=CIFAR10_N_CLASSES,
                       size=CIFAR10_SIZE,

                       dataset_class=CIFAR10_DATASET,

                       train_transform=CIFAR10_TRAIN_TRANSFORM,
                       train_percentage=CIFAR10_TRAIN_PERCENTAGE,

                       val_transform=CIFAR10_VALIDATION_TRANSFORM,
                       val_percentage=CIFAR10_VALIDATION_PERCENTAGE,

                       test_transform=CIFAR10_TEST_TRANSFORM,
                       test_percentage=CIFAR10_TEST_PERCENTAGE,

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

        self._train_val_length = CIFAR10_TRAIN_VAL_LENGTH
        self._test_val_length = CIFAR10_TEST_LENGTH

        self._train_indices = None
        self._val_indices = None

    def prepare_data(self):
        # download
        self._dataset_class(self._path, train=True, download=True, transform=None)
        self._dataset_class(self._path, train=False, download=True, transform=None)

    def reset_indices(self):
        train_n_indices = math.floor(self._train_percentage * self._train_val_length)
        val_n_indices = math.ceil(self._val_percentage * self._train_val_length)
        train_val_indices = randperm(cifar10_train_n_indices + cifar10_val_n_indices).tolist()
        self._train_indices = train_val_indices[0:train_n_indices]
        self._val_indices = train_val_indices[train_n_indices:(train_n_indices + val_n_indices)]


    def setup(self, stage):
        if self._train_indices is None or self._val_indices is None:
            self.reset_indices()
        cifar10_train = self._dataset_class(self._path, train=True, download=False, transform=self._train_transform)
        cifar10_val = self._dataset_class(self._path, train=True, download=False, transform=self._val_transform)
        cifar10_test = self._dataset_class(self._path, train=False, download=False, transform=self._test_transform)

        self._train_dataset = torch.utils.data.Subset(cifar10_train, self._train_indices)
        self._val_dataset = torch.utils.data.Subset(cifar10_val, self._val_indices)
        self._test_dataset = cifar10_test

DATASET = {CIFAR10_NAME: CIFAR10DataModule}
