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

import pathlib
import sys

import pytorch_lightning

try:
    import dvs128gesturesnnmodule
    import dvs128gesturedatamodule
except ImportError:
    sys.path.append(str(pathlib.Path(__file__).absolute().parent))
    import dvs128gesturesnnmodule
    import dvs128gesturedatamodule

    sys.path.pop()

BATCH_SIZE = 10
DVS128GESTURE_DATASET_PATH = pathlib.Path(
    "/shared/ml/datasets/vision/snn/DVS128Gesture/"
)
MONITOR_METRIC_ACCURACY = "val_acc_epoch"
MONITOR_METRIC_ACCURACY_MODE = "max"
MONITOR_METRIC_LOSS = "val_loss_epoch"
MONITOR_METRIC_LOSS_MODE = "min"
# MONITOR_METRIC_LOSS = "val_acc_epoch"
# MONITOR_METRIC_LOSS_MODE = "max"
SEED = 420
TRAINING_DIR = pathlib.Path(__file__).parent / "checkpoints" / "dvs128gesture_snn"


def main():
    pytorch_lightning.seed_everything(SEED)

    model = dvs128gesturesnnmodule.DVS128GestureSNNModule(
        encoder=dvs128gesturesnnmodule.DVS128GestureSNNModule.encoder_dvs128gesture,
        decoder=dvs128gesturesnnmodule.DVS128GestureSNNModule.decoder_dvs128gesture,
        return_state=False,
        encoding_flag=True,
        decoding_flag=True,
        trainable_neuron_parameters=False,
        learning_rate=1e-3,
    )
    datamodule = dvs128gesturedatamodule.DVS128GestureDataModule(
        data_dir=DVS128GESTURE_DATASET_PATH,
        num_workers=64,
        drop_last=False,
        shuffle=False,
        batch_size=BATCH_SIZE,
        seed=SEED,
        pin_memory=False,
    )

    trainer = pytorch_lightning.Trainer(
        accelerator="gpu",
        callbacks=[
            pytorch_lightning.callbacks.DeviceStatsMonitor(),
            pytorch_lightning.callbacks.EarlyStopping(
                check_finite=True,
                min_delta=0.001,
                mode=MONITOR_METRIC_LOSS_MODE,
                # string of monitored metric
                # default is early_stop_on
                monitor=MONITOR_METRIC_LOSS,
                patience=5,
                verbose=True,
            ),
            pytorch_lightning.callbacks.ModelCheckpoint(
                dirpath=None,
                every_n_epochs=1,
                every_n_train_steps=None,
                filename=None,
                mode=MONITOR_METRIC_ACCURACY_MODE,
                monitor=MONITOR_METRIC_ACCURACY,
                save_last=True,
                save_top_k=3,
                save_weights_only=False,
                verbose=True,
            ),
            pytorch_lightning.callbacks.TQDMProgressBar(),
        ],
        default_root_dir=str(TRAINING_DIR),
        deterministic=True,
        devices="auto",
        logger=[
            pytorch_lightning.loggers.TensorBoardLogger(
                save_dir=str(TRAINING_DIR),
                # experiment name, in this custom configuration it is default
                name="default",
                version=None,
                # this enables the saving of the computational graph
                # it requires example_input_array in the model
                log_graph=True,
                default_hp_metric=True,
                prefix="",
            )
        ],
        log_every_n_steps=10,
        replace_sampler_ddp=True,
        strategy=pytorch_lightning.plugins.DDPPlugin(find_unused_parameters=False),
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
