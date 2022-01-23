# -*- coding: utf-8 -*-
import pathlib
import typing

import flash.image

import enpheeph
import enpheeph.injections.plugins.mask.autopytorchmaskplugin

import base_config
import cifar10_config
import quantization_config


def config(
    *,
    dataset_directory: pathlib.Path,
    model_weight_file: pathlib.Path,
    storage_file: pathlib.Path,
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    storage_plugin = enpheeph.injections.plugins.storage.SQLiteStoragePlugin(
        db_url="sqlite:///" + str(storage_file)
    )
    pytorch_mask_plugin = enpheeph.injections.plugins.mask.AutoPyTorchMaskPlugin()
    pytorch_handler_plugin = enpheeph.handlers.plugins.PyTorchHandlerPlugin()

    monitor_1 = enpheeph.injections.OutputPyTorchMonitor(
        location=enpheeph.utils.data_classes.MonitorLocation(
            module_name="adapter.backbone.conv1",
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
            module_name="adapter.backbone.conv1",
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
            module_name="adapter.backbone.conv1",
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
                enpheeph.utils.enums.DimensionType.Tensor: (slice(10, 100),),
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

    injection_handler = enpheeph.handlers.InjectionHandler(
        injections=[monitor_1, fault_1, monitor_2, monitor_3, fault_2, monitor_4],
        library_handler_plugin=pytorch_handler_plugin,
    )
    injection_handler.activate()

    model = {
        "callable": flash.image.classification.ImageClassifier.load_from_checkpoint,
        "callable_args": {
            "checkpoint_path": str(model_weight_file),
            # issues with loading GPU model on CPU
            # it should work with PyTorch but there must be some problems with
            # PyTorch Lightning/Flash leading to use some GPU memory
            "map_location": "cpu",
        },
    }

    config = base_config.config()
    # datamodule update
    config.update(cifar10_config.config(dataset_directory=dataset_directory))
    # dynamic quantization update
    config.update(quantization_config.config())
    config["model"] = model
    # update the Trainer with flash as we are using flash models, to avoid
    # compatibility issues such as CUDA out of memory on CPU-only
    config["trainer"]["callable"] = flash.Trainer

    # we delay the instantiation of the callback to allow the saving of the
    # current configuration
    callback = enpheeph.integrations.pytorchlightning.InjectionCallback(
        injection_handler=injection_handler,
        storage_plugin=storage_plugin,
        extra_session_info=config,
    )

    config["trainer"]["callable_args"]["callbacks"].append(callback)

    # to save the injection handler to enable/disable faults
    config["injection_handler"] = injection_handler

    return config
