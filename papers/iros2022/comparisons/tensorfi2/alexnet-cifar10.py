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
import sys
import typing

import flash
import flash.image
import pytorch_lightning
import torch
import torchmetrics
import torchvision

import enpheeph
import enpheeph.injections.plugins.indexing.indexingplugin


CURRENT_DIR = pathlib.Path(__file__).absolute().parent
RESULTS_DIRECTORY = CURRENT_DIR / "results" / "alexnet-cifar10"
WEIGHTS_FILE = RESULTS_DIRECTORY / "weights" / "alexnet-cifar10.pt"
LOG_DIRECTORY = RESULTS_DIRECTORY / "injection_results"

WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

CIFAR_DIRECTORY = pathlib.Path("/shared/ml/datasets/vision/") / "CIFAR10"


class AlexNetLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, pretrained: bool = True, num_classes: int = 1000) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained

        self.model = torchvision.models.AlexNet(num_classes=num_classes)
        if self.pretrained:
            # must be accessed with sys.modules otherwise it uses the function
            # which is imported from the sub-module
            # we use type: ignore as mypy cannot check torchvision typings
            # we have to split it otherwise black creates problems
            mod = sys.modules["torchvision.models.alexnet"]
            state_dict = torch.hub.load_state_dict_from_url(
                mod.model_urls["alexnet"],  # type: ignore[attr-defined]
                progress=True,
            )
            # we must filter the mismatching keys in the state dict
            # we generate the current model state dict
            model_state_dict = self.model.state_dict()
            filtered_state_dict = {
                k: (
                    v_new
                    # we select the new value if the dimension is the same as with the old
                    # one
                    if v_new.size() == v_old.size()
                    # otherwise we use the initialized one from the model
                    else v_old
                )
                for (k, v_old), v_new in zip(
                    model_state_dict.items(),
                    state_dict.values(),
                )
            }

            self.model.load_state_dict(filtered_state_dict, strict=False)

        self.normalizer_fn = torch.nn.Softmax(dim=-1)
        self.accuracy_fn = torchmetrics.Accuracy()
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

        # we initialize the weights
        self.init_weights()

    def init_weights(self) -> None:
        # this initialization is similar to the ResNet one
        # taken from https://github.com/Lornatang/AlexNet-PyTorch/
        # @ alexnet_pytorch/model.py#L63
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        return self.model(inpt)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-2)
        return optimizer

    def inference(
        self,
        batch: typing.Union[
            torch.Tensor,
            typing.Dict[flash.core.data.data_source.DefaultDataKeys, torch.Tensor],
        ],
        batch_idx: int,
    ) -> typing.Dict[str, torch.Tensor]:
        # we need to check for the batch to be a flash batch or to be a standard tuple
        # as otherwise it may not be compatible
        if isinstance(batch, dict):
            x = batch.get(flash.core.data.data_source.DefaultDataKeys.INPUT, None)
            y = batch.get(flash.core.data.data_source.DefaultDataKeys.TARGET, None)
            if x is None or y is None:
                raise ValueError("Incompatible input for the batch")
        else:
            x, y = batch

        output = self.forward(x)
        return {
            "loss": self.loss_fn(output, y),
            "accuracy": self.accuracy_fn(self.normalizer_fn(output), y),
        }

    def training_step(
        self,
        batch: typing.Union[
            torch.Tensor,
            typing.Dict[flash.core.data.data_source.DefaultDataKeys, torch.Tensor],
        ],
        batch_idx: int,
    ) -> torch.Tensor:
        res = self.inference(batch, batch_idx)
        self.log_dict(
            {"train_loss": res["loss"], "train_accuracy": res["accuracy"]},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return res["loss"]

    def validation_step(
        self,
        batch: typing.Union[
            torch.Tensor,
            typing.Dict[flash.core.data.data_source.DefaultDataKeys, torch.Tensor],
        ],
        batch_idx: int,
    ) -> None:
        res = self.inference(batch, batch_idx)
        self.log_dict(
            {"val_loss": res["loss"], "val_accuracy": res["accuracy"]},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

    def test_step(
        self,
        batch: typing.Union[
            torch.Tensor,
            typing.Dict[flash.core.data.data_source.DefaultDataKeys, torch.Tensor],
        ],
        batch_idx: int,
    ) -> None:
        res = self.inference(batch, batch_idx)
        self.log_dict(
            {"test_loss": res["loss"], "test_accuracy": res["accuracy"]},
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )


pytorch_lightning.seed_everything(seed=41, workers=True)


storage_plugin = enpheeph.injections.plugins.storage.SQLiteStoragePlugin(
    db_url="sqlite:///" + str(LOG_DIRECTORY / "database.sqlite")
)
pytorch_mask_plugin = enpheeph.injections.plugins.NumPyPyTorchMaskPlugin()
pytorch_handler_plugin = enpheeph.handlers.plugins.PyTorchHandlerPlugin()


monitor_1 = enpheeph.injections.OutputPyTorchMonitor(
    location=enpheeph.utils.data_classes.MonitorLocation(
        module_name="model.features.0",
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
    indexing_plugin=enpheeph.injections.plugins.indexing.indexingplugin.IndexingPlugin(
        dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
    ),
)
fault_1 = enpheeph.injections.OutputPyTorchFault(
    location=enpheeph.utils.data_classes.FaultLocation(
        module_name="model.features.0",
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
    indexing_plugin=enpheeph.injections.plugins.indexing.indexingplugin.IndexingPlugin(
        dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
    ),
)
monitor_2 = enpheeph.injections.OutputPyTorchMonitor(
    location=enpheeph.utils.data_classes.MonitorLocation(
        module_name="model.features.0",
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
    indexing_plugin=enpheeph.injections.plugins.indexing.indexingplugin.IndexingPlugin(
        dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
    ),
)
monitor_3 = enpheeph.injections.OutputPyTorchMonitor(
    location=enpheeph.utils.data_classes.MonitorLocation(
        module_name="model.classifier.1",
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
    indexing_plugin=enpheeph.injections.plugins.indexing.indexingplugin.IndexingPlugin(
        dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
    ),
)
fault_2 = enpheeph.injections.OutputPyTorchFault(
    location=enpheeph.utils.data_classes.FaultLocation(
        module_name="model.classifier.1",
        parameter_type=enpheeph.utils.enums.ParameterType.Activation,
        dimension_index={
            enpheeph.utils.enums.DimensionType.Tensor: (slice(10, 100),),
            enpheeph.utils.enums.DimensionType.Batch: ...,
        },
        bit_index=...,
        bit_fault_value=enpheeph.utils.enums.BitFaultValue.StuckAtOne,
    ),
    low_level_torch_plugin=pytorch_mask_plugin,
    indexing_plugin=enpheeph.injections.plugins.indexing.indexingplugin.IndexingPlugin(
        dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
    ),
)
monitor_4 = enpheeph.injections.OutputPyTorchMonitor(
    location=enpheeph.utils.data_classes.MonitorLocation(
        module_name="model.classifier.1",
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
    indexing_plugin=enpheeph.injections.plugins.indexing.indexingplugin.IndexingPlugin(
        dimension_dict=enpheeph.utils.constants.PYTORCH_DIMENSION_DICT,
    ),
)

injection_handler = enpheeph.handlers.InjectionHandler(
    injections=[monitor_1, fault_1, monitor_2, monitor_3, fault_2, monitor_4],
    library_handler_plugin=pytorch_handler_plugin,
)

callback = enpheeph.integrations.pytorchlightning.InjectionCallback(
    injection_handler=injection_handler,
    storage_plugin=storage_plugin,
)

trainer = pytorch_lightning.Trainer(
    callbacks=[callback],
    deterministic=True,
    enable_checkpointing=False,
    max_epochs=10,
    # one can use gpu but some functions will not be deterministic, so deterministic
    # must be set to False
    accelerator="cpu",
    devices=1,
    # if one uses spawn or dp it will fail as sqlite connector is not picklable
    # strategy="ddp",
)

model = AlexNetLightningModule(num_classes=10, pretrained=False)

# transform = torchvision.transforms.Compose(
#     [
#         #torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize(
#             (0.5, 0.5, 0.5),
#             (0.5, 0.5, 0.5),
#         ),
#         torchvision.transforms.RandomHorizontalFlip(),
#     ]
# )
cifar_train = torchvision.datasets.CIFAR10(
    str(CIFAR_DIRECTORY),
    train=True,
    download=True,
)
cifar_test = torchvision.datasets.CIFAR10(
    str(CIFAR_DIRECTORY),
    train=False,
    download=True,
)
datamodule = flash.image.ImageClassificationData.from_datasets(
    train_dataset=cifar_train,
    test_dataset=cifar_test,
    val_split=0.2,
    num_workers=64,
    batch_size=32,
)

if not WEIGHTS_FILE.exists():
    trainer.fit(
        model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader(),
    )

    trainer.save_checkpoint(str(WEIGHTS_FILE))

model = model.load_from_checkpoint(str(WEIGHTS_FILE))

# no injections/monitors
print("\n\nBaseline, no injection or monitors\n")
trainer.test(
    model,
    dataloaders=datamodule.test_dataloader(),
)

# we enable only the monitors
# we use this as baseline, no injections
callback.injection_handler.activate([monitor_1, monitor_2, monitor_3, monitor_4])
print("\n\nBaseline, no injection, only monitors\n")
trainer.test(
    model,
    dataloaders=datamodule.test_dataloader(),
)

# we enable the faults
callback.injection_handler.activate([fault_1, fault_2])
print("\n\nWeight + activation injection\n")
trainer.test(
    model,
    dataloaders=datamodule.test_dataloader(),
)


# we disable the faults
callback.injection_handler.deactivate([fault_1, fault_2])
print("\n\nBaseline again, no injection, only monitors\n")
# we test again to reach same results as before injection
trainer.test(
    model,
    dataloaders=datamodule.test_dataloader(),
)
