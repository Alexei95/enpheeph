# -*- coding: utf-8 -*-
import pathlib
import typing

import flash
import torchvision


def config(
    dataset_directory: pathlib.Path,
) -> typing.Dict[str, typing.Any]:
    return {
        "datamodule": {
            "callable": flash.image.ImageClassificationData.from_datasets,
            "callable_args": {
                "train_dataset": torchvision.datasets.CIFAR10(
                    str(dataset_directory),
                    train=True,
                    download=True,
                ),
                "test_dataset": torchvision.datasets.CIFAR10(
                    str(dataset_directory),
                    train=False,
                    download=True,
                ),
                "val_split": 0.2,
                "num_workers": 64,
                "batch_size": 32,
            },
        },
    }
