import abc
import copy
import pathlib
import sys

import pytorch_lightning as pl
import torch
import torchvision.datasets
import torchvision.transforms

from ...common import DEFAULT_DATASET_PATH
from .common import *


# metaclass usage for abstract class definition
# or inheritance-based abstract class
class BaseDataModule(pl.DataModule, abc.ABC):
    _name = None
    _n_classes = None
    _size = None

    def __init__(self, path=DEFAULT_DATASET_PATH,

                       name=None,
                       n_classes=None,
                       size=None,

                       dataset_class=None,

                       train_transform=DEFAULT_TRAIN_TRANSFORM,
                       train_normalize=DEFAULT_TRAIN_NORMALIZE,
                       train_batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                       train_percentage=DEFAULT_TRAIN_PERCENTAGE,
                       train_shuffle=DEFAULT_TRAIN_SHUFFLE,
                       train_num_workers=DEFAULT_TRAIN_NUM_WORKERS,
                       train_pin_memory=DEFAULT_TRAIN_PIN_MEMORY,
                       train_drop_incomplete_batch=DEFAULT_TRAIN_DROP_INCOMPLETE_BATCH,
                       train_batch_timeout=DEFAULT_TRAIN_BATCH_TIMEOUT,

                       val_transform=DEFAULT_VALIDATION_TRANSFORM,
                       val_normalize=DEFAULT_VALIDATION_NORMALIZE,
                       val_batch_size=DEFAULT_VALIDATION_BATCH_SIZE,
                       val_percentage=DEFAULT_VALIDATION_PERCENTAGE,
                       val_shuffle=DEFAULT_VALIDATION_SHUFFLE,
                       val_num_workers=DEFAULT_VALIDATION_NUM_WORKERS,
                       val_pin_memory=DEFAULT_VALIDATION_PIN_MEMORY,
                       val_drop_incomplete_batch=DEFAULT_VALIDATION_DROP_INCOMPLETE_BATCH,
                       val_batch_timeout=DEFAULT_VALIDATION_BATCH_TIMEOUT,

                       test_transform=DEFAULT_TEST_TRANSFORM,
                       test_normalize=DEFAULT_TEST_NORMALIZE,
                       test_batch_size=DEFAULT_TEST_BATCH_SIZE,
                       test_percentage=DEFAULT_TEST_PERCENTAGE,
                       test_shuffle=DEFAULT_TEST_SHUFFLE,
                       test_num_workers=DEFAULT_TEST_NUM_WORKERS,
                       test_pin_memory=DEFAULT_TEST_PIN_MEMORY,
                       test_drop_incomplete_batch=DEFAULT_TEST_DROP_INCOMPLETE_BATCH,
                       test_batch_timeout=DEFAULT_TEST_BATCH_TIMEOUT,

                       *args,
                       **kwargs):
        super().__init__(*args, **kwargs)

        self._path = pathlib.Path(path).resolve()
        self._name = name
        self._n_classes = n_classes
        self._size = size
        self._dataset_class = dataset_class

        self._train_transform = self.setup_transform(transform=train_transform,
                                                     normalize=train_normalize,
                                                     default=DEFAULT_TRAIN_TRANSFORM)
        self._train_normalize = train_normalize
        self._train_percentage = train_percentage
        self._train_batch_size = train_batch_size
        self._train_shuffle = train_shuffle
        self._train_num_workers = train_num_workers
        self._train_pin_memory = train_pin_memory
        self._train_drop_incomplete_batch = train_drop_incomplete_batch
        self._train_batch_timeout = train_batch_timeout
        self._train_dataset = None

        self._val_transform = self.setup_transform(transform=val_transform,
                                                     normalize=val_normalize,
                                                     default=DEFAULT_VALIDATION_TRANSFORM)
        self._val_normalize = val_normalize
        self._val_percentage = val_percentage
        self._val_batch_size = val_batch_size
        self._val_shuffle = val_shuffle
        self._val_num_workers = val_num_workers
        self._val_pin_memory = val_pin_memory
        self._val_drop_incomplete_batch = val_drop_incomplete_batch
        self._val_batch_timeout = val_batch_timeout
        self._val_dataset = None

        self._test_transform = self.setup_transform(transform=test_transform,
                                                     normalize=test_normalize,
                                                     default=DEFAULT_TEST_TRANSFORM)
        self._test_normalize = test_normalize
        self._test_batch_size = test_batch_size
        self._test_percentage = test_percentage
        self._test_shuffle = test_shuffle
        self._test_num_workers = test_num_workers
        self._test_pin_memory = test_pin_memory
        self._test_drop_incomplete_batch = test_drop_incomplete_batch
        self._test_batch_timeout = test_batch_timeout
        self._test_dataset = None

        self._base_asserts()

    def _asserts(self, *args, **kwargs):
        pass

    def _base_asserts(self):
        '''This function contains all the basic assertions which are run after
        receiving the arguments.
        '''
        for dataset_type in ['train', 'val', 'test']:
            assert getattr(self, f"_{dataset_type}_percentage") > 0
            assert getattr(self, f"_{dataset_type}_percentage") < 1

    @property
    def path(self):
        return self._path

    @property
    def name(self):
        return self._name

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def size(self):
        return self._size

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
        if self._train_dataset is None:
            raise Exception('train dataset must be initialized')
        return torch.utils.data.DataLoader(self._train_dataset,
                                           batch_size=self._train_batch_size,
                                           num_workers=self._train_num_workers,
                                           pin_memory=self._train_pin_memory,
                                           shuffle=self._train_shuffle,
                                           drop_last=self._train_drop_incomplete_batch,
                                           timeout=self._train_batch_timeout)

    def val_dataloader(self):
        if self._val_dataset is None:
            raise Exception('validation dataset must be initialized')
        return torch.utils.data.DataLoader(self._val_dataset,
                                           batch_size=self._val_batch_size,
                                           num_workers=self._val_num_workers,
                                           pin_memory=self._val_pin_memory,
                                           shuffle=self._val_shuffle,
                                           drop_last=self._val_drop_incomplete_batch,
                                           timeout=self._val_batch_timeout)

    def test_dataloader(self):
        if self._test_dataset is None:
            raise Exception('test dataset must be initialized')
        return torch.utils.data.DataLoader(self._test_dataset,
                                           batch_size=self._test_batch_size,
                                           num_workers=self._test_num_workers,
                                           pin_memory=self._test_pin_memory,
                                           shuffle=self._test_shuffle,
                                           drop_last=self._test_drop_incomplete_batch,
                                           timeout=self._test_batch_timeout)
