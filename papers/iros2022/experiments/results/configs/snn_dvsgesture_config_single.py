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

import enpheeph.injections
import enpheeph.injections.plugins

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
    storage_plugin = config["injection_callback"].storage_plugin
    pytorch_mask_plugin = (
        enpheeph.injections.plugins.mask.autopytorchmaskplugin.AutoPyTorchMaskPlugin()
    )

    fault_2 = enpheeph.injections.OutputPyTorchFault(
        location=enpheeph.utils.data_classes.FaultLocation(
            # 2/6 is conv, 11/13 is linear
            # 3 is lif
            module_name="sequential.2",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: (slice(10, 15), ...),
                enpheeph.utils.enums.DimensionType.Batch: ...,
                enpheeph.utils.enums.DimensionType.Time: ...,
            },
            bit_index=[31],
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.StuckAtOne,
        ),
        low_level_torch_plugin=pytorch_mask_plugin,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.NORSE_DIMENSION_DICT,
        ),
    )

    config["injection_handler"].add_injections(
        injections=[fault_2],
    )

    # custom is used to avoid the random injections
    config["injection_config"] = {
        "custom": True,
    }

    return config
