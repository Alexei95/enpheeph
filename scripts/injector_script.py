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

import importlib
import pathlib
import sys
import typing

import pytorch_lightning

import enpheeph
import enpheeph.injections.abc
import enpheeph.injections.pruneddensetosparseactivationpytorchfault


def get_injection_callback() -> pytorch_lightning.Callback:
    storage_file = (
        pathlib.Path(__file__).absolute().parent
        / "results/injection_test/database.sqlite"
    )
    storage_file.parent.mkdir(exist_ok=True, parents=True)

    pytorch_handler_plugin = enpheeph.handlers.plugins.PyTorchHandlerPlugin()
    storage_plugin = enpheeph.injections.plugins.storage.SQLiteStoragePlugin(
        db_url="sqlite:///" + str(storage_file)
    )
    pytorch_mask_plugin = enpheeph.injections.plugins.mask.AutoPyTorchMaskPlugin()

    fault_1 = enpheeph.injections.PrunedDenseToSparseWeightPyTorchFault(
        location=enpheeph.utils.dataclasses.FaultLocation(
            # resnet18
            module_name="adapter.backbone.layer1.0.conv1",
            parameter_type=(
                enpheeph.utils.enums.ParameterType.Weight
                | enpheeph.utils.enums.ParameterType.Sparse
                | enpheeph.utils.enums.ParameterType.Value
            ),
            parameter_name="weight",
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: (0,),
            },
            bit_index=...,
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.StuckAtOne,
        ),
        low_level_torch_plugin=pytorch_mask_plugin,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )
    monitor_1 = enpheeph.injections.OutputPyTorchMonitor(
        location=enpheeph.utils.dataclasses.MonitorLocation(
            module_name="adapter.head",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: ...,
                enpheeph.utils.enums.DimensionType.Batch: ...,
            },
            bit_index=None,
        ),
        enabled_metrics=(
            enpheeph.utils.enums.MonitorMetric.ArithmeticMean
            | enpheeph.utils.enums.MonitorMetric.StandardDeviation
        ),
        storage_plugin=storage_plugin,
        move_to_first=False,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )

    injection_handler = enpheeph.handlers.InjectionHandler(
        injections=[fault_1, monitor_1],
        library_handler_plugin=pytorch_handler_plugin,
    )

    # we delay the instantiation of the callback to allow the saving of the
    # current configuration
    callback = enpheeph.integrations.pytorchlightning.InjectionCallback(
        injection_handler=injection_handler,
        storage_plugin=storage_plugin,
        # this config used to contain the complete system config: trainer + model +
        # dataset, including the configuration for injections
        # extra_session_info=config,
    )
    return callback


def get_trainer_config(args=sys.argv) -> typing.Dict[str, typing.Any]:
    config = pathlib.Path(args[1]).absolute()

    sys.path.append(str(config.parent))

    module_name = config.with_suffix("").name

    module = importlib.import_module(module_name)

    sys.path.pop()

    config_dict = module.config()

    return config_dict


def main():
    config = pathlib.Path(sys.argv[1]).absolute()

    sys.path.append(str(config.parent))

    module_name = config.with_suffix("").name

    module = importlib.import_module(module_name)

    sys.path.pop()

    config_dict = module.config()

    trainer = config_dict["trainer"]
    datamodule = config_dict["datamodule"]
    model = config_dict["model"]

    injection_callback = get_injection_callback()
    trainer.callbacks.append(injection_callback)

    injection_callback.injection_handler.activate()
    injection_callback.injection_handler.deactivate(
        [
            inj
            for inj in injection_callback.injection_handler.injections
            if isinstance(inj, enpheeph.injections.abc.FaultABC)
        ]
    )
    # print(injection_callback.injection_handler.active_injections)
    trainer.test(model, datamodule=datamodule)

    injection_callback.injection_handler.activate()
    # print(injection_callback.injection_handler.active_injections)
    trainer.test(model, datamodule=datamodule)

    injection_callback.injection_handler.deactivate(
        [
            inj
            for inj in injection_callback.injection_handler.injections
            if isinstance(inj, enpheeph.injections.abc.FaultABC)
        ]
    )
    # print(injection_callback.injection_handler.active_injections)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
