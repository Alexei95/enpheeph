# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2023 Alessio "Alexei95" Colucci
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

import enpheeph.injections
import enpheeph.injections.plugins

import random_multi_config
import snn_dvsgesture_config


def config(
    *,
    dataset_directory: pathlib.Path,
    model_weight_file: pathlib.Path,
    storage_file: pathlib.Path,
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    config = snn_dvsgesture_config.config(
        dataset_directory=dataset_directory,
        model_weight_file=model_weight_file,
        storage_file=storage_file,
    )
    config.update(random_multi_config.config())

    # custom is used to avoid the random injections
    config["injection_config"]["layers"] = [
        # only conv2d, linear is not working
        "sequential.2",
        "sequential.6",
        # linear does not work yet
        # "sequential.11",
        # "sequential.13"
    ]
    config["injection_config"][
        "indexing_dimension_dict"
    ] = enpheeph.utils.constants.NORSE_DIMENSION_DICT

    return config
