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

DEFAULT_RESULT_DIRECTORY = (
    "results/inference_pruned_semantic_segmenter_mobilenetv3_carla"
)

# DEFAULT_MODEL_CHECKPOINT = (
#     "results/pruned_image_classifier_resnet18_cifar10/default/"
#     "version_9/checkpoints/epoch=64-step=81250.ckpt"
# )
# val acc 0.798 val loss 0.642 pruning 31%
# DEFAULT_MODEL_CHECKPOINT = (
#     "results/pruned_image_classifier_resnet18_cifar10/default/"
#     "version_11/checkpoints/epoch=19-step=25000.ckpt"
# )
# val acc 0.711 val loss 0.837 pruning 71.9%
DEFAULT_MODEL_CHECKPOINT = (
    "results/pruned_semantic_segmenter_mobilenetv3_carla_new/default/"
    "version_0/checkpoints/epoch=16-step=850.ckpt"
)


def config(
    *,
    dataset_directory: os.PathLike = "/shared/ml/datasets/vision/",
    result_directory: os.PathLike = DEFAULT_RESULT_DIRECTORY,
    checkpoint_file: os.PathLike = DEFAULT_MODEL_CHECKPOINT,
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    pytorch_lightning.seed_everything(seed=42, workers=True)

    full_dataset_train_directory = (
        pathlib.Path(dataset_directory)
        / "carla-data-capture/20180528-100vehicles-100pedestrians/CameraRGB/"
    )
    full_dataset_train_target_directory = (
        pathlib.Path(dataset_directory)
        / "carla-data-capture/20180528-100vehicles-100pedestrians/CameraSeg/"
    )
    full_result_directory = pathlib.Path(result_directory)

    datamodule = flash.image.SemanticSegmentationData.from_folders(
        batch_size=16,
        # image_size=(256, 256),
        num_classes=101,
        num_workers=64,
        test_folder=str(full_dataset_train_directory),
        test_target_folder=str(full_dataset_train_target_directory),
        train_folder=str(full_dataset_train_directory),
        train_target_folder=str(full_dataset_train_target_directory),
        val_split=0.2,
    )

    # model = flash.image.SemanticSegmentation(
    #     backbone="mobilenetv3_large_100",
    #     head="fpn",
    #     learning_rate=0.001,
    #     num_classes=101,
    #     optimizer="Adam",
    #     pretrained=True,
    # )
    model = flash.image.SemanticSegmentation.load_from_checkpoint(checkpoint_file)

    trainer = flash.Trainer(
        accelerator="auto",
        auto_lr_find=False,
        auto_scale_batch_size=False,
        benchmark=False,
        callbacks=[
            pytorch_lightning.callbacks.DeviceStatsMonitor(),
            pytorch_lightning.callbacks.TQDMProgressBar(
                refresh_rate=10,
            ),
        ],
        check_val_every_n_epoch=1,
        default_root_dir=str(full_result_directory),
        detect_anomaly=False,
        deterministic=False,
        devices="auto",
        enable_checkpointing=True,
        enable_model_summary=True,
        enable_progress_bar=True,
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
