# -*- coding: utf-8 -*-
import pathlib

# import sys

import pytorch_lightning

import enpheeph.extra.injectioncallback
import enpheeph.handlers.injectionhandler
import enpheeph.handlers.plugins.pytorchhandlerplugin
import enpheeph.injections.outputpytorchfault
import enpheeph.injections.outputpytorchmonitor

# not ready
# import enpheeph.injections.snnoutputnorsefault
import enpheeph.injections.plugins.cupypytorchmaskplugin
import enpheeph.injections.plugins.numpypytorchmaskplugin
import enpheeph.injections.plugins.picklestorageplugin
import enpheeph.utils.data_classes
import enpheeph.utils.enums

import pynndora.models.vision.lenet5
import pynndora.datasets.vision.mnist

FILE_PATH = pathlib.Path(__file__).absolute()
FILE_DIR = FILE_PATH.parent.absolute()
ENPHEEPH_PATH = FILE_DIR.parent.absolute()
PYNNDORA_PATH = (ENPHEEPH_PATH.parent / "pynndora").absolute()
# if str(ENPHEEPH_PATH) not in sys.path:
#     sys.path.append(str(ENPHEEPH_PATH))
# if str(PYNNDORA_PATH) not in sys.path:
#     sys.path.append(str(PYNNDORA_PATH))


MODEL_PATH = (
    PYNNDORA_PATH
    / "model_training/lightning/lenet5_mnist"
    / "default/version_5/checkpoints/epoch=25-step=9749.ckpt"
)

storage_plugin = enpheeph.injections.plugins.picklestorageplugin.PickleStoragePlugin(
    path=FILE_DIR / "test.pickle",
)
pytorch_mask_plugin = (
    enpheeph.injections.plugins.cupypytorchmaskplugin.CuPyPyTorchMaskPlugin()
)
pytorch_handler_plugin = (
    enpheeph.handlers.plugins.pytorchhandlerplugin.PyTorchHandlerPlugin()
)


monitor_1 = enpheeph.injections.outputpytorchmonitor.OutputPyTorchMonitor(
    monitor_location=enpheeph.utils.data_classes.InjectionLocation(
        module_name="feature_extractor.0",
        parameter_type=enpheeph.utils.enums.ParameterType.Activation,
        tensor_index=...,
        bit_index=...,
        time_index=None,
    ),
    enabled_metrics=enpheeph.utils.enums.MonitorMetric.StandardDeviation,
    storage_plugin=storage_plugin,
    move_to_first=False,
)
fault_1 = enpheeph.injections.outputpytorchfault.OutputPyTorchFault(
    fault_location=enpheeph.utils.data_classes.FaultLocation(
        module_name="feature_extractor.0",
        parameter_type=enpheeph.utils.enums.ParameterType.Activation,
        tensor_index=...,
        bit_index=...,
        time_index=None,
        bit_fault_value=enpheeph.utils.enums.BitFaultValue.StuckAtOne,
    ),
    low_level_torch_plugin=pytorch_mask_plugin,
)
monitor_2 = enpheeph.injections.outputpytorchmonitor.OutputPyTorchMonitor(
    monitor_location=enpheeph.utils.data_classes.InjectionLocation(
        module_name="feature_extractor.0",
        parameter_type=enpheeph.utils.enums.ParameterType.Activation,
        tensor_index=...,
        bit_index=...,
        time_index=None,
    ),
    enabled_metrics=enpheeph.utils.enums.MonitorMetric.StandardDeviation,
    storage_plugin=storage_plugin,
    move_to_first=False,
)

injection_handler = enpheeph.handlers.injectionhandler.InjectionHandler(
    injections=[monitor_1, fault_1, monitor_2],
    library_handler_plugin=pytorch_handler_plugin,
)

callback = enpheeph.extra.injectioncallback.InjectionCallback(
    injection_manager=injection_handler, storage_plugin=storage_plugin,
)

trainer = pytorch_lightning.Trainer(gpus=[2], callbacks=[callback])
datamodule = pynndora.datasets.vision.mnist.MNISTDataModule(
    batch_size=4,
    data_dir="/shared/ml/datasets/vision/MNIST/",
    drop_last=True,
    num_workers=0,
    pin_memory=False,
    seed=42,
    shuffle=False,
)
# we load the model from the checkpoint
# we use strict=False as there are some extra info, since the model is not
# fully initialized yet
# then we initialize the model based on the init configuration using the
# lazy_model_init_handler
model = pynndora.models.vision.lenet5.LeNet5.load_from_checkpoint(
    str(MODEL_PATH), strict=False,
)
model.lazy_model_init_handler()

print(trainer.test(model, datamodule))

injection_handler.activate()

print(trainer.test(model, datamodule))
