import math
import pathlib
import sys

import pytorch_lightning as pl
import pl_bolts
import torch
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

from . import visiondatamoduleabc


### COMPLETE IMAGENET IMPLEMENTATION

IMAGENET_TRAIN_VAL_LENGTH = 50000
IMAGENET_TEST_LENGTH = 10000
IMAGENET_TRAIN_PERCENTAGE = 0.9
IMAGENET_N_VAL_IMAGES_PER_CLASS = 10
IMAGENET_TEST_PERCENTAGE = 1.0
IMAGENET_DATASET = pl_bolts.datamodules.imagenet_dataset.UnlabeledImagenet
IMAGENET_NAME = IMAGENET_DATASET.__name__
IMAGENET_SIZE = torch.Size([3, 227, 227])
IMAGENET_N_CLASSES = 1000
IMAGENET_TRAIN_TRANSFORM = torchvision.transforms.Compose([
                            # check how to implement this
                            # torchvision.transforms.RandomResizedCrop(self.image_size),
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            ),
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


class ImageNetDataModule(visiondatamoduleabc.VisionDataModuleABC):
    _name = IMAGENET_NAME
    _n_classes = IMAGENET_N_CLASSES
    _size = IMAGENET_SIZE

    def __init__(self, name=IMAGENET_NAME,
                       n_classes=IMAGENET_N_CLASSES,
                       size=IMAGENET_SIZE,

                       dataset_class=IMAGENET_DATASET,

                       train_transform=IMAGENET_TRAIN_TRANSFORM,
                       train_percentage=IMAGENET_TRAIN_PERCENTAGE,

                       val_transform=IMAGENET_VALIDATION_TRANSFORM,
                       n_val_images_per_class=IMAGENET_N_VAL_IMAGES_PER_CLASS,

                       test_transform=IMAGENET_TEST_TRANSFORM,
                       test_percentage=IMAGENET_TEST_PERCENTAGE,

                       *args,
                       **kwargs):
        kwargs.update({'name': name,
                       'dataset_class': dataset_class,
                       'train_transform': train_transform,
                       'train_percentage': train_percentage,
                       'val_transform': val_transform,
                       'val_percentage': val_percentage,
                       'test_transform': test_transform,
                       'test_percentage': test_percentage, })
        super().__init__(*args, **kwargs)

        return NotImplemented

        self._train_val_length = CIFAR10_TRAIN_VAL_LENGTH
        self._test_val_length = CIFAR10_TEST_LENGTH

        self._train_indices = None
        self._val_indices = None

    def prepare_data(self):
        return NotImplemented
        # download
        self._dataset_class(self._path, train=True, download=True, transform=None)
        self._dataset_class(self._path, train=False, download=True, transform=None)

    def reset_indices(self):
        return NotImplemented
        train_n_indices = math.floor(self._train_percentage * self._train_val_length)
        val_n_indices = math.ceil(self._val_percentage * self._train_val_length)
        train_val_indices = torch.randperm(train_n_indices + val_n_indices).tolist()
        self._train_indices = train_val_indices[0:train_n_indices]
        self._val_indices = train_val_indices[train_n_indices:(train_n_indices + val_n_indices)]

    def setup(self, stage):
        return NotImplemented
        if self._train_indices is None or self._val_indices is None:
            self.reset_indices()
        cifar10_train = self._dataset_class(self._path, train=True, download=False, transform=self._train_transform)
        cifar10_val = self._dataset_class(self._path, train=True, download=False, transform=self._val_transform)
        cifar10_test = self._dataset_class(self._path, train=False, download=False, transform=self._test_transform)

        self._train_dataset = torch.utils.data.Subset(cifar10_train, self._train_indices)
        self._val_dataset = torch.utils.data.Subset(cifar10_val, self._val_indices)
        self._test_dataset = cifar10_test
