# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2024 Alessio "Alexei95" Colucci
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

import flash.image


def config(
    *,
    dataset_directory: pathlib.Path,
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    return {
        "datamodule": {
            "callable": flash.image.SemanticSegmentationData.from_folders,
            "callable_args": {
                "batch_size": 32,
                "image_size": [256, 256],
                "num_classes": 101,
                "num_workers": 64,
                "test_folder": str(
                    dataset_directory
                    / "carla-data-capture/20180528-100vehicles-100pedestrians/CameraRGB/"
                ),
                "test_target_folder": str(
                    dataset_directory
                    / "carla-data-capture/20180528-100vehicles-100pedestrians/CameraSeg/"
                ),
            },
        },
    }
