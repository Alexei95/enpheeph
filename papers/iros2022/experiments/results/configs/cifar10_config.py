# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pathlib
import typing

import flash
import torchvision


def config(
    *,
    dataset_directory: pathlib.Path,
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    return {
        "datamodule": {
            "callable": flash.image.ImageClassificationData.from_datasets,
            "callable_args": {
                "train_dataset": torchvision.datasets.CIFAR10(
                    str(dataset_directory / "CIFAR10"),
                    train=True,
                    download=True,
                ),
                "test_dataset": torchvision.datasets.CIFAR10(
                    str(dataset_directory / "CIFAR10"),
                    train=False,
                    download=True,
                ),
                "val_split": 0.2,
                "num_workers": 64,
                "batch_size": 32,
            },
        },
    }
