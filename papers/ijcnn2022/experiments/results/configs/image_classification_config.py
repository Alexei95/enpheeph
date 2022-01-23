# -*- coding: utf-8 -*-
import pathlib
import typing

import flash.image

import enpheeph

import base_config
import cifar10_config
import quantization_config


def config(
    dataset_directory: pathlib.Path,
    results_directory: pathlib.Path,
) -> typing.Dict[str, typing.Any]:
    model_file = pathlib.Path("weights/image_classification.pt")
    complete_model_file = results_directory / model_file

    storage_plugin = enpheeph.injections.plugins.storage.SQLiteStoragePlugin(
        db_url="sqlite:///"
        + str(results_directory / model_file.with_suffix(".sqlite").name)
    )
    pytorch_mask_plugin = enpheeph.injections.plugins.NumPyPyTorchMaskPlugin()
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
    fault_1 = enpheeph.injections.OutputPyTorchFault(
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
                enpheeph.utils.enums.DimensionType.Batch: ...,
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
            module_name="adapter.head.0",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: (slice(10, 100),),
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
            module_name="adapter.head.0",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: (slice(10, 100),),
                enpheeph.utils.enums.DimensionType.Batch: ...,
            },
            bit_index=...,
            bit_fault_value=enpheeph.utils.enums.BitFaultValue.StuckAtOne,
        ),
        low_level_torch_plugin=pytorch_mask_plugin,
        indexing_plugin=enpheeph.injections.plugins.indexing.IndexingPlugin(
            dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
        ),
    )
    monitor_4 = enpheeph.injections.OutputPyTorchMonitor(
        location=enpheeph.utils.data_classes.MonitorLocation(
            module_name="adapter.head.0",
            parameter_type=enpheeph.utils.enums.ParameterType.Activation,
            dimension_index={
                enpheeph.utils.enums.DimensionType.Tensor: (slice(10, 100),),
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

    callback = enpheeph.integrations.pytorchlightning.InjectionCallback(
        injection_handler=injection_handler,
        storage_plugin=storage_plugin,
    )

    model = {
        "callable": flash.image.classification.ImageClassifier.load_from_checkpoint,
        "callable_args": {
            "checkpoint_path": str(complete_model_file),
            # issues with loading GPU model on CPU
            # it should work with PyTorch but there must be some problems with
            # PyTorch Lightning/Flash leading to use some GPU memory
            "map_location": "cuda",
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
    config["trainer"]["callable_args"]["callbacks"].append(callback)

    # to save the injection handler to enable/disable faults
    config["injection_handler"] = injection_handler

    return config