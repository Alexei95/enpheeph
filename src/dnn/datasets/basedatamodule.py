import copy
import pathlib
import sys

import pytorch_lightning as pl
import torch
import torchvision.datasets
import torchvision.transforms

from ...common import DEFAULT_DATASET_PATH
from .common import *


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, path=DEFAULT_DATASET_PATH,

                       train_transform=None,
                       train_normalize=DEFAULT_TRAIN_NORMALIZE,
                       train_batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                       train_percentage=DEFAULT_TRAIN_PERCENTAGE,

                       val_transform=None,
                       val_normalize=DEFAULT_VAL_NORMALIZE,
                       val_batch_size=DEFAULT_VALIDATION_BATCH_SIZE,
                       val_percentage=DEFAULT_VALIDATION_PERCENTAGE,

                       test_transform=None,
                       test_normalize=DEFAULT_TEST_NORMALIZE,
                       test_batch_size=DEFAULT_TEST_BATCH_SIZE,
                       test_percentage=DEFAULT_TEST_PERCENTAGE,

                       *args,
                       **kwargs):
        super().__init__(*args, **kwargs)

        self._path = pathlib.Path(path).resolve()

        self._train_transform = self.setup_transform(transform=train_transform,
                                                     normalize=train_normalize,
                                                     default=DEFAULT_TRAIN_TRANSFORM)
        self._train_normalize = train_normalize
        self._train_percentage = train_percentage
        self._train_batch_size = train_batch_size

        self._val_transform = self.setup_transform(transform=val_transform,
                                                     normalize=val_normalize,
                                                     default=DEFAULT_VALIDATION_TRANSFORM)
        self._val_normalize = val_normalize
        self._val_percentage = val_percentage
        self._val_batch_size = val_batch_size

        self._test_transform = self.setup_transform(transform=test_transform,
                                                     normalize=test_normalize,
                                                     default=DEFAULT_TEST_TRANSFORM)
        self._test_normalize = test_normalize
        self._test_batch_size = test_batch_size
        self._test_percentage = test_percentage

    @staticmethod
    def setup_transform(transform=None, normalize=DEFAULT_NORMALIZE, default=DEFAULT_TRANSFORM):
        if transform is None:
            return copy.deepcopy(default)
        else:
            transform = copy.deepcopy(transform)
            if not normalize and hasattr(transform, 'transforms'):
                transform.transforms = [t for t in transform.transforms if not isinstance(t, torchvision.transforms.Normalize)]
        return transform

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=64)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=64)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=64)
