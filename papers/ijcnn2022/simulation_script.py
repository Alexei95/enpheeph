# -*- coding: utf-8 -*-
# type: ignore
# flake: noqa
# we skip flake and mypy as
import pathlib

# import sys

import pytorch_lightning

import enpheeph

# import enpheeph.handlers.injectionhandler
# import enpheeph.handlers.plugins.pytorchhandlerplugin
# import enpheeph.injections.outputpytorchfault
# import enpheeph.injections.outputpytorchmonitor
# # not ready
# # import enpheeph.injections.snnoutputnorsefault
# import enpheeph.injections.plugins.mask.cupypytorchmaskplugin
# import enpheeph.injections.plugins.mask.numpypytorchmaskplugin
# import enpheeph.injections.plugins.storage.sqlstorageplugin.sqlitestorageplugin
# import enpheeph.integrations.pytorchlightning.injectioncallback
# import enpheeph.utils.data_classes
# import enpheeph.utils.enums

FILE_PATH = pathlib.Path(__file__).absolute()
FILE_DIR = FILE_PATH.parent.absolute()
ENPHEEPH_PATH = FILE_DIR.parent.absolute()
PYNNDORA_PATH = (ENPHEEPH_PATH.parent / "pynndora").absolute()
# if str(ENPHEEPH_PATH) not in sys.path:
#     sys.path.append(str(ENPHEEPH_PATH))
# if str(PYNNDORA_PATH) not in sys.path:
#     sys.path.append(str(PYNNDORA_PATH))
# MODEL_PATH = (
#     PYNNDORA_PATH
#     / "model_training/lightning/lenet5_mnist"
#     / "default/version_5/checkpoints/epoch=25-step=9749.ckpt"
# )

storage_plugin = enpheeph.SQLiteStoragePlugin(
    db_url=":memory:",
    engine_echo_debug=True,
)
pytorch_mask_plugin = enpheeph.CuPyPyTorchMaskPlugin()
pytorch_handler_plugin = enpheeph.PyTorchHandlerPlugin()


monitor_1 = enpheeph.OutputPyTorchMonitor(
    location=enpheeph.MonitorLocation(
        module_name="feature_extractor.0",
        parameter_type=enpheeph.ParameterType.Activation,
        tensor_index=...,
        bit_index=...,
        time_index=None,
    ),
    enabled_metrics=enpheeph.MonitorMetric.StandardDeviation,
    storage_plugin=storage_plugin,
    move_to_first=False,
)
fault_1 = enpheeph.OutputPyTorchFault(
    location=enpheeph.FaultLocation(
        module_name="feature_extractor.0",
        parameter_type=enpheeph.ParameterType.Activation,
        tensor_index=...,
        bit_index=...,
        time_index=None,
        bit_fault_value=enpheeph.BitFaultValue.StuckAtOne,
    ),
    low_level_torch_plugin=pytorch_mask_plugin,
)
monitor_2 = enpheeph.OutputPyTorchMonitor(
    location=enpheeph.MonitorLocation(
        module_name="feature_extractor.0",
        parameter_type=enpheeph.ParameterType.Activation,
        tensor_index=...,
        bit_index=...,
        time_index=None,
    ),
    enabled_metrics=enpheeph.MonitorMetric.StandardDeviation,
    storage_plugin=storage_plugin,
    move_to_first=False,
)

injection_handler = enpheeph.InjectionHandler(
    injections=[monitor_1, fault_1, monitor_2],
    library_handler_plugin=pytorch_handler_plugin,
)

callback = enpheeph.InjectionCallback(
    injection_handler=injection_handler,
    storage_plugin=storage_plugin,
)

trainer = pytorch_lightning.Trainer(gpus=[2], callbacks=[callback])


# print(trainer.test(model, datamodule))

injection_handler.activate()

# print(trainer.test(model, datamodule))

# print(
#     storage_plugin.get_experiment_ids(
#         [monitor_1.location, fault_1.location, monitor_2.location]
#     )
# )
