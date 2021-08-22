import copy
import functools
import pathlib

import typing

import pl_bolts
import pytorch_lightning
import tonic
import tonic.datasets.ncars
import torch

import src.ml.pl.datasets.utils.monkeypatchedtonicdataset
import src.ml.pl.datasets.vision.snn.neuromorphicdatamodule


class NCARSDataModule(
        src.ml.pl.datasets.vision.snn.neuromorphicdatamodule.
        NeuromorphicDataModule
):

    dataset_cls = src.ml.pl.datasets.utils.\
        monkeypatchedtonicdataset.monkey_patching_tonic_dataset(
                tonic.datasets.ncars.NCARS,
                'ncars-train',
                'ncars-test',
        )
    name = "N-CARS"
    dims = (2, 120, 100)

    def default_transforms(self) -> typing.Callable:
        return tonic.transforms.Compose()
