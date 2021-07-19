import copy
import functools
import pathlib

import typing

import pl_bolts
import pytorch_lightning
import tonic
import tonic.datasets.nmnist
import torch

import src.ml.pl.datasets.utils.monkeypatchedtonicdataset
import src.ml.pl.datasets.vision.snn.neuromorphicdatamodule


class NMNISTDataModule(
        src.ml.pl.datasets.vision.snn.neuromorphicdatamodule.
        NeuromorphicDataModule
):

    EXTRA_ARGS = (
            src.ml.pl.datasets.vision.snn.neuromorphicdatamodule.
            NeuromorphicDataModule.EXTRA_ARGS | {
                    'first_saccade_only': False
            }
    )
    dataset_cls = src.ml.pl.datasets.utils.\
        monkeypatchedtonicdataset.monkey_patching_tonic_dataset(
                tonic.datasets.nmnist.NMNIST,
                tonic.datasets.nmnist,
                tonic.datasets.nmist.NMNIST.train_filename,
                tonic.datasets.nmist.NMNIST.test_filename,
        )
    name = "N-MNIST"
    dims = (2, 34, 34)

    def __init__(
            self,
            first_saccade_only: bool = False,
            *args: typing.Any,
            **kwargs: typing.Any,
    ):
        super().__init__(
                *args,
                **kwargs
        )

        # this is automatically passed in the dataset class
        self.EXTRA_ARGS['first_saccade_only'] = self.first_saccade_only

    def default_transforms(self) -> typing.Callable:
        return tonic.transforms.Compose()
