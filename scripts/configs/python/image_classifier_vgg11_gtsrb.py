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

import os
import pathlib
import typing

import flash
import flash.image
import pytorch_lightning
import torchvision


def compute_pruning_amount(epoch):
    # return 0.02
    # if epoch <= 5:
    #     return 0.5
    # elif epoch <= 10:
    #     return 0.2
    # else:
    #     return 0.01
    if epoch < 30:
        return 0
    else:  # if 30 <= epoch:
        if epoch % 5 == 0:
            return 0.05
        else:
            return 0


def config(
    *,
    dataset_directory: os.PathLike = "/shared/ml/datasets/vision/",
    result_directory: os.PathLike = "results/image_classifier_vgg11_gtsrb",
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    pytorch_lightning.seed_everything(seed=42, workers=True)

    full_dataset_directory = pathlib.Path(dataset_directory) / "GTSRB"
    full_result_directory = pathlib.Path(result_directory)

    # we create the two datasets with the data
    train_dataset = torchvision.datasets.GTSRB(
        str(full_dataset_directory),
        split="train",
        download=True,
    )
    test_dataset = torchvision.datasets.GTSRB(
        str(full_dataset_directory),
        split="test",
        download=True,
    )
    datamodule = flash.image.ImageClassificationData.from_datasets(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=32,
        num_workers=64,
        transform_kwargs={
            "image_size": (48, 48),
            # "mean": (0.485, 0.456, 0.406),
            # "std": (0.229, 0.224, 0.225)
        },
        val_split=0.2,
    )

    model = flash.image.ImageClassifier(
        backbone="vgg11",
        learning_rate=0.001,
        num_classes=43,
        pretrained=True,
    )

    trainer = flash.Trainer(
        accelerator="auto",
        auto_lr_find=False,
        auto_scale_batch_size=False,
        benchmark=False,
        callbacks=[
            pytorch_lightning.callbacks.DeviceStatsMonitor(),
            pytorch_lightning.callbacks.EarlyStopping(
                check_finite=True,
                check_on_train_epoch_end=False,
                divergence_threshold=None,
                min_delta=0.01,
                mode="min",
                monitor="val_cross_entropy",
                patience=30,
                stopping_threshold=None,
                strict=True,
                verbose=True,
            ),
            pytorch_lightning.callbacks.LearningRateMonitor(
                log_momentum=True,
                logging_interval="epoch",
            ),
            pytorch_lightning.callbacks.ModelCheckpoint(
                dirpath=None,
                every_n_epochs=1,
                every_n_train_steps=None,
                filename=None,
                mode="min",
                # otherwise saving at the end of the training epoch would not have a
                # proper validation loss available
                monitor="val_cross_entropy",
                save_last=True,
                # to save the checkpoint at the end of training epoch
                # instead of validation epoch
                save_on_train_epoch_end=False,
                save_top_k=-1,
                # save_top_k=3,
                save_weights_only=False,
                verbose=True,
            ),
            # pytorch_lightning.callbacks.ModelPruning(
            #     amount=compute_pruning_amount,
            #     apply_pruning=True,
            #     make_pruning_permanent=True,
            #     parameter_names=("weight", "bias"),
            #     parameters_to_prune=None,
            #     # we need to prune at the end of validation (using False here)
            #     # as the checkpoint is saved at the end of training
            #     prune_on_train_epoch_end=True,
            #     pruning_dim=None,
            #     # pruning_fn="random_unstructured",
            #     pruning_fn="l1_unstructured",
            #     pruning_norm=None,
            #     resample_parameters=True,
            #     use_global_unstructured=True,
            #     use_lottery_ticket_hypothesis=True,
            #     verbose=True,
            # ),
            # pytorch_lightning.callbacks.TQDMProgressBar(
            #     refresh_rate=1000000,
            # ),
        ],
        check_val_every_n_epoch=1,
        default_root_dir=str(full_result_directory),
        detect_anomaly=False,
        deterministic=True,
        devices="auto",
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=False,
        fast_dev_run=False,
        gradient_clip_algorithm=None,
        gradient_clip_val=None,
        ipus=None,
        log_every_n_steps=10,
        logger=[
            pytorch_lightning.loggers.TensorBoardLogger(
                default_hp_metric=True,
                log_graph=True,
                name="default",
                prefix="",
                save_dir=str(full_result_directory),
                version=None,
            ),
        ],
        max_epochs=-1,
        max_steps=-1,
        max_time=None,
        min_epochs=1,
        min_steps=None,
        move_metrics_to_cpu=False,
        multiple_trainloader_mode="min_size",
        num_nodes=1,
        num_processes=1,
        num_sanity_val_steps=2,
        plugins=[],
        precision=32,
        profiler=None,
        reload_dataloaders_every_n_epochs=0,
        replace_sampler_ddp=True,
        strategy=None,
        sync_batchnorm=False,
        tpu_cores=None,
        track_grad_norm=-1,
        weights_save_path=None,
    )

    return {"trainer": trainer, "model": model, "datamodule": datamodule}
