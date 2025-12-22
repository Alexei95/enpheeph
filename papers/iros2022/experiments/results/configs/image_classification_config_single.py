# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2025 Alessio "Alexei95" Colucci
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

import image_classification_config


def config(
    *,
    dataset_directory: pathlib.Path,
    model_weight_file: pathlib.Path,
    storage_file: pathlib.Path,
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    config = image_classification_config.config(
        dataset_directory=dataset_directory,
        model_weight_file=model_weight_file,
        storage_file=storage_file,
    )
    storage_plugin = config["injection_callback"].storage_plugin
    pytorch_mask_plugin = (
        enpheeph.injections.plugins.mask.autopytorchmaskplugin.AutoPyTorchMaskPlugin()
    )

    monitor_1 = enpheeph.injections.OutputPyTorchMonitor(
        location=enpheeph.utils.data_classes.MonitorLocation(
            # resnet18
            # module_name="adapter.backbone.conv1",
            # vgg11
            module_name="adapter.backbone.0",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: ...,
                enpheeph.utils.enums.DimensionType.Batch: ...,
            },
            bit_index=...,
        ),
        enabled_metrics=enpheeph.utils.enums.MonitorMetric.StandardDeviation,
        storage_plugin=storage_plugin,
        move_to_first=False,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )
    fault_1 = enpheeph.injections.WeightPyTorchFault(
        location=enpheeph.utils.data_classes.FaultLocation(
            # resnet18
            # module_name="adapter.backbone.conv1",
            # vgg11
            module_name="adapter.backbone.0",
            parameter_type=enpheeph.utils.enums.ParameterType.Weight,
            parameter_name="weight",
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: (
                    ...,
                    0,
                    0,
                ),
            },
            bit_index=[10, 16, 31],
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.StuckAtOne,
        ),
        low_level_torch_plugin=pytorch_mask_plugin,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )
    monitor_2 = enpheeph.injections.OutputPyTorchMonitor(
        location=enpheeph.utils.data_classes.MonitorLocation(
            # resnet18
            # module_name="adapter.backbone.conv1",
            # vgg11
            module_name="adapter.backbone.0",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: ...,
                enpheeph.utils.enums.DimensionType.Batch: ...,
            },
            bit_index=None,
        ),
        enabled_metrics=enpheeph.utils.enums.MonitorMetric.StandardDeviation,
        storage_plugin=storage_plugin,
        move_to_first=False,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )
    monitor_3 = enpheeph.injections.OutputPyTorchMonitor(
        location=enpheeph.utils.data_classes.MonitorLocation(
            module_name="adapter.backbone",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: ...,
                enpheeph.utils.enums.DimensionType.Batch: ...,
            },
            bit_index=None,
        ),
        enabled_metrics=enpheeph.utils.enums.MonitorMetric.StandardDeviation,
        storage_plugin=storage_plugin,
        move_to_first=False,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )
    fault_2 = enpheeph.injections.OutputPyTorchFault(
        location=enpheeph.utils.data_classes.FaultLocation(
            module_name="adapter.backbone",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: (slice(10, 15),),
                enpheeph.utils.enums.DimensionType.Batch: ...,
            },
            bit_index=[31],
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.StuckAtOne,
        ),
        low_level_torch_plugin=pytorch_mask_plugin,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )
    monitor_4 = enpheeph.injections.OutputPyTorchMonitor(
        location=enpheeph.utils.data_classes.MonitorLocation(
            module_name="adapter.backbone",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: ...,
                enpheeph.utils.enums.DimensionType.Batch: ...,
            },
            bit_index=None,
        ),
        enabled_metrics=enpheeph.utils.enums.MonitorMetric.StandardDeviation,
        storage_plugin=storage_plugin,
        move_to_first=False,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )

    config["injection_handler"].add_injections(
        injections=[monitor_1, fault_1, monitor_2, monitor_3, fault_2, monitor_4],
    )

    # custom is used to avoid the random injections
    config["injection_config"] = {
        "custom": True,
    }

    return config
