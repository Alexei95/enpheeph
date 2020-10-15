import abc
import copy
import pathlib
import sys

import torch
import torchvision.datasets
import torchvision.transforms

from ....common import DEFAULT_DATASET_PATH
from ..common import *
from .common import *
from .. import datamoduleabc


# metaclass usage for abstract class definition
# or inheritance-based abstract class
# no abstract class as it creates metaclass issues
class VisionDataModuleABC(datamoduleabc.DataModuleABC):
    _n_classes = None

    def __init__(self, n_classes=None,

                       train_transform=VISION_DEFAULT_TRAIN_TRANSFORM,

                       val_transform=VISION_DEFAULT_VALIDATION_TRANSFORM,

                       test_transform=VISION_DEFAULT_TRAIN_TRANSFORM,

                       *args,
                       **kwargs):

        # to add missing arguments to kwargs, in this way we can have different defaults
        kwargs['train_transform'] = train_transform
        kwargs['val_transform'] = val_transform
        kwargs['test_transform'] = test_transform

        super().__init__(*args, **kwargs)

        self._n_classes = n_classes

    @classmethod
    def n_classes(cls):
        return cls._n_classes

    @staticmethod
    def setup_transform(transform=None, normalize=VISION_DEFAULT_NORMALIZE, default=VISION_DEFAULT_TRANSFORM):
        if transform is None:
            return copy.deepcopy(default)
        else:
            transform = copy.deepcopy(transform)
            if not normalize and hasattr(transform, 'transforms'):
                transform.transforms = [t for t in transform.transforms if not isinstance(t, torchvision.transforms.Normalize)]
        return transform
