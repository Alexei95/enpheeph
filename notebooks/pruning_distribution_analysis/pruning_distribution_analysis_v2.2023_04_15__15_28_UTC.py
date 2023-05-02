#!/usr/bin/env python
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

# # Use the enpheeph-dev mamba environment
# The old one is enpheeph-dev-old-lightning-flash

# In[1]:


import math
import os
import pathlib
import time

import captum
import lightning
import numpy
import pandas
import torch
import torch.optim
import torchmetrics
import torchvision
import torchvision.datasets
import torchvision.transforms


# In[2]:


class Model(lightning.LightningModule):
    LAYER_LIST = {
        "vgg11": {
            "model.features.0": [64, 32, 32],
            "model.features.1": [64, 32, 32],
            "model.features.2": [64, 16, 16],
            "model.features.3": [128, 16, 16],
            "model.features.4": [128, 16, 16],
            "model.features.5": [128, 8, 8],
            "model.features.6": [256, 8, 8],
            "model.features.7": [256, 8, 8],
            "model.features.8": [256, 8, 8],
            "model.features.9": [256, 8, 8],
            "model.features.10": [256, 4, 4],
            "model.features.11": [512, 4, 4],
            "model.features.12": [512, 4, 4],
            "model.features.13": [512, 4, 4],
            "model.features.14": [512, 4, 4],
            "model.features.15": [512, 2, 2],
            "model.features.16": [512, 2, 2],
            "model.features.17": [512, 2, 2],
            "model.features.18": [512, 2, 2],
            "model.features.19": [512, 2, 2],
            "model.features.20": [512, 1, 1],
            "model.avgpool": [512, 7, 7],
            "model.classifier.0": [4096],
            "model.classifier.1": [4096],
            "model.classifier.2": [4096],
            "model.classifier.3": [4096],
            "model.classifier.4": [4096],
            "model.classifier.5": [4096],
            "model.classifier.6": [10],
        },
    }

    def __init__(
        self,
        model_name,
        num_classes,
        accuracy_fn,
        loss_fn,
        dataframe_path,
        optimizer_class,
        learning_rate,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.num_classes = num_classes
        self.accuracy = accuracy_fn
        self.loss = loss_fn
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate

        self.dataframe_path = pathlib.Path(dataframe_path)

        self.setup_model(model_name=self.model_name, num_classes=self.num_classes)

        self.handles = []

        self.reset_dataframe()
        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def reset_dataframe(self):
        self.dataframe = pandas.DataFrame(
            columns=[
                "module_name",
                "tensor_type",
                "batch_index",
                "element_in_batch_index",
                "location",
                "value",
                "accuracy",
                "loss",
            ]
        )

    @staticmethod
    def join_saved_dataframe(dataframe, dataframe_path: os.PathLike):
        dataframe_path = pathlib.Path(dataframe_path)
        if not dataframe_path.exists():
            dataframe_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_csv(dataframe_path, sep="|")
        else:
            df = pandas.read_csv(dataframe_path, sep="|", index_col=[0], header=[0])
            new_df = pandas.concat([df, dataframe], axis=0)
            new_df.reset_index(drop=True, inplace=True)
            new_df.to_csv(dataframe_path, sep="|")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        return optimizer

    def make_neuron_output_function(self, module_name, location):
        def save_neuron_output(module, args, output) -> None:
            for b_idx, b in enumerate(output):
                self.dataframe.loc[len(self.dataframe)] = [
                    module_name,
                    "output",
                    None,
                    b_idx,
                    location,
                    b[location].item(),
                    None,
                    None,
                ]

        return save_neuron_output

    def add_hooks(self, attributions, topk=1):
        for layer_name, layer_attributions_and_deltas in attributions.items():
            layer_attributions_cat = torch.cat(
                tuple(l_attr for l_attr, _ in layer_attributions_and_deltas),
                dim=0,
            )
            summed_layer_attributions = torch.sum(
                layer_attributions_cat,
                (0,),
            )
            topk_values, topk_indices = torch.topk(
                abs(
                    summed_layer_attributions.flatten(),
                ),
                k=topk,
                largest=True,
                sorted=True,
            )
            for top_index in topk_indices:
                target_neuron_location = numpy.unravel_index(
                    top_index,
                    summed_layer_attributions.size(),
                    order="C",
                )
                module = self.get_layer_from_full_name(
                    self,
                    layer_name,
                    separator=".",
                    main_model_is_in_the_layer_name=False,
                )
                self.handles.append(
                    module.register_forward_hook(
                        self.make_neuron_output_function(
                            layer_name, tuple(target_neuron_location)
                        )
                    )
                )

    def setup_model(self, model_name, num_classes):
        if model_name == "vgg11":
            self.model = torchvision.models.vgg11(
                num_classes=num_classes, init_weights=True
            )
        elif model_name == "mlp":
            self.model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, num_classes),
            )
        else:
            raise ValueError("unknown model")

    @staticmethod
    def get_full_layer_name_from_summary(layer_summary, skip_main_model=True):
        parent_info = layer_summary.parent_info
        layer_full_name = layer_summary.var_name
        while parent_info is not None and (
            not skip_main_model
            or skip_main_model
            and parent_info.parent_info is not None
        ):
            layer_full_name = f"{parent_info.var_name}.{layer_full_name}"
            parent_info = parent_info.parent_info
        return layer_full_name

    @staticmethod
    def get_layer_from_full_name(
        model, layer_name, separator=".", main_model_is_in_the_layer_name=False
    ):
        module = model
        if main_model_is_in_the_layer_name:
            layer_name = separator.join(layer_name.split(separator)[1:])
        for l_n in layer_name.split(separator):
            module = getattr(module, l_n)
        return module

    def get_attributions(
        self,
        dataloader,
        layer_name_list,
        attributions_checkpoint_path,
        attribution=captum.attr.LayerConductance,
        save_checkpoint=True,
        load_checkpoint=True,
    ):
        if attributions_checkpoint_path.exists() and load_checkpoint:
            attributions = torch.load(str(attributions_checkpoint_path))
            return attributions
        elif save_checkpoint:
            attributions_checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

        model = self.train(False).to(torch.device("cpu"))

        attributions = {}
        for layer_name in layer_name_list:
            print(layer_name)
            layer_attributions = []
            attr_instance = attribution(
                model, model.get_layer_from_full_name(model, layer_name)
            )
            for idx, b in enumerate(dataloader):
                x, y = b
                attr, delta = attr_instance.attribute(
                    inputs=x.to(torch.device("cpu")),
                    target=y.to(torch.device("cpu")),
                    return_convergence_delta=True,
                )
                layer_attributions.append(
                    [
                        attr.detach(),
                        delta.detach(),
                    ],
                )
                if idx % 10 == 0:
                    print(f"Batches done: {idx}")
            attributions[layer_name] = layer_attributions
            if save_checkpoint:
                torch.save(attributions, str(attributions_checkpoint_path))

        if save_checkpoint:
            torch.save(attributions, str(attributions_checkpoint_path))

        return attributions

    def inference_step(self, batch, only_x=False):
        if only_x:
            x = batch
        else:
            x, y = batch
        y_hat = self(x)
        if only_x:
            d = {"loss": None, "accuracy": None, "predictions": y_hat}
        else:
            d = {
                "loss": self.loss(y_hat, y),
                "accuracy": self.accuracy(y_hat, y),
                "predictions": y_hat,
            }
        return d

    def training_step(self, batch, batch_idx):
        metrics = self.inference_step(batch)
        self.log_dict(
            {"train_loss": metrics["loss"], "train_accuracy": metrics["accuracy"]},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return metrics["loss"]

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.inference_step(batch)
        self.log_dict(
            {"test_loss": metrics["loss"], "test_accuracy": metrics["accuracy"]},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return metrics["predictions"]

    def validation_step(self, batch, batch_idx):
        metrics = self.inference_step(batch)
        self.log_dict(
            {"val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return metrics["predictions"]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        metrics = self.inference_step(batch, only_x=True)
        # self.log({"val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics["predictions"]

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)
        _, y = batch
        row_selector = (
            self.dataframe["accuracy"].isnull() & self.dataframe["loss"].isnull()
        )
        self.dataframe.loc[row_selector, "batch_index"] = batch_idx
        assert (
            len(self.dataframe.loc[row_selector]) / len(self.handles)
            == y.size()[0]
            == outputs.size()[0]
        )
        for bindex, (by_hat, by) in enumerate(zip(outputs, y)):
            by_hat = by_hat.unsqueeze(0)
            by = by.unsqueeze(0)
            extra_row_selector = row_selector & (
                self.dataframe["element_in_batch_index"] == bindex
            )
            self.dataframe.loc[extra_row_selector, "loss"] = self.loss(
                by_hat, by
            ).item()
            self.dataframe.loc[extra_row_selector, "accuracy"] = self.accuracy(
                by_hat, by
            ).item()
        self.dataframe_path.parent.mkdir(parents=True, exist_ok=True)
        if batch_idx % 10 == 0:
            self.join_saved_dataframe(self.dataframe, self.dataframe_path)
            self.reset_dataframe()
        # print(self.dataframe)

    def on_test_end(self):
        self.join_saved_dataframe(self.dataframe, self.dataframe_path)


class DataModule(lightning.LightningDataModule):
    MNIST_DEFAULT_TRANSFORM = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,),
                (0.3081,),
            ),
        ]
    )
    CIFAR10_DEFAULT_TRANSFORM = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    def __init__(
        self,
        dataset_class,
        data_dir: str = "/shared/ml/datasets/vision/",
        train_transform=None,
        test_transform=None,
        batch_size=64,
        num_workers=32,
        train_val_split=0.8,
        seed=42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.num_workers = num_workers
        self.seed = seed

        if issubclass(dataset_class, torchvision.datasets.MNIST):
            if train_transform is None:
                train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomCrop(28, padding=4),
                        self.MNIST_DEFAULT_TRANSFORM,
                    ]
                )
            if test_transform is None:
                test_transform = self.MNIST_DEFAULT_TRANSFORM
            self.num_classes = 10
        elif issubclass(dataset_class, torchvision.datasets.CIFAR10):
            if train_transform is None:
                train_transform = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.RandomCrop(32, padding=4),
                        torchvision.transforms.RandomHorizontalFlip(),
                        self.CIFAR10_DEFAULT_TRANSFORM,
                    ]
                )
            if test_transform is None:
                test_transform = self.CIFAR10_DEFAULT_TRANSFORM
            self.num_classes = 10
        else:
            raise ValueError("unknown dataset")

        self.dataset_class = dataset_class

        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        # download
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dataset_train_transform = self.dataset_class(
                self.data_dir, train=True, transform=self.train_transform
            )
            n_train_elements = math.floor(
                len(dataset_train_transform) * self.train_val_split
            )
            self.dataset_train, _ = torch.utils.data.random_split(
                dataset_train_transform,
                [n_train_elements, len(dataset_train_transform) - n_train_elements],
                generator=torch.Generator().manual_seed(self.seed),
            )
            dataset_test_transform = self.dataset_class(
                self.data_dir, train=True, transform=self.test_transform
            )
            _, self.dataset_val = torch.utils.data.random_split(
                dataset_test_transform,
                [n_train_elements, len(dataset_train_transform) - n_train_elements],
                generator=torch.Generator().manual_seed(self.seed),
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.dataset_test = self.dataset_class(
                self.data_dir, train=False, transform=self.test_transform
            )

        if stage == "predict":
            self.dataset_predict = self.dataset_class(
                self.data_dir, train=False, transform=self.test_transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# In[3]:


TIME_FORMAT = "%Y_%m_%d__%H_%M_%S_%z"

time_string = time.strftime(TIME_FORMAT)

base_path = pathlib.Path("./results/trained_vgg11_cifar10_earlystopping_lightning")
model_checkpoint_path = base_path.with_suffix(f".{time_string}.pt")
attributions_checkpoint_path = base_path.with_suffix(f".{time_string}.attributions.pt")
dataframe_path = base_path.with_suffix(f".{time_string}.csv")

learning_rate_finder = False
seed = 7

lightning.seed_everything(seed)


# In[4]:


trainer = lightning.Trainer(
    accelerator="gpu",
    devices=[2],
    max_epochs=-1,
    callbacks=[
        lightning.pytorch.callbacks.EarlyStopping(
            "val_loss",
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            stopping_threshold=None,
            divergence_threshold=None,
            check_on_train_epoch_end=None,
            log_rank_zero_only=False,
        ),
        lightning.pytorch.callbacks.ModelCheckpoint(
            dirpath=None,
            filename=None,
            monitor=None,
            verbose=False,
            save_last=None,
            # to disable model saving
            save_top_k=0,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=True,
            every_n_train_steps=None,
            train_time_interval=None,
            every_n_epochs=None,
            save_on_train_epoch_end=None,
        ),
        lightning.pytorch.callbacks.RichProgressBar(
            refresh_rate=10,
        ),
        lightning.pytorch.callbacks.StochasticWeightAveraging(
            swa_lrs=1e-2,
        ),
    ],
)

model = Model(
    model_name="vgg11",
    num_classes=10,
    accuracy_fn=torchmetrics.Accuracy(
        task="multiclass",
        num_classes=10,
    ),
    loss_fn=torch.nn.CrossEntropyLoss(),
    dataframe_path=dataframe_path,
    optimizer_class=torch.optim.AdamW,
    learning_rate=1e-3,
)
datamodule = DataModule(
    dataset_class=torchvision.datasets.CIFAR10,
    data_dir="/shared/ml/datasets/vision/CIFAR10",
    train_transform=None,
    test_transform=None,
    batch_size=64,
    train_val_split=0.8,
    seed=seed,
)

tuner = lightning.pytorch.tuner.Tuner(trainer)


# In[5]:


if learning_rate_finder:
    tuner.lr_find(model, datamodule=datamodule)
    print(model.learning_rate)

if model_checkpoint_path.exists():
    model.__class__.load_from_checkpoint(str(model_checkpoint_path))
else:
    model_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.fit(model, datamodule)
    trainer.save_checkpoint(str(model_checkpoint_path))


# In[ ]:


datamodule.prepare_data()
datamodule.setup(stage="test")

attributions = model.__class__.get_attributions(
    model,
    datamodule.test_dataloader(),
    list(model.LAYER_LIST[model.model_name].keys()),
    attributions_checkpoint_path=attributions_checkpoint_path,
    save_checkpoint=True,
    load_checkpoint=True,
)

datamodule.teardown(stage="test")

model.add_hooks(attributions, topk=3)


# In[ ]:


trainer.test(model, datamodule, ckpt_path=str(model_checkpoint_path))


# In[ ]:


# dataframe_path = model_checkpoint_path.with_suffix(f".{time.strftime(TIME_FORMAT)}.csv")
# model.dataframe.to_csv(dataframe_path, sep="|")
# model.dataframe
