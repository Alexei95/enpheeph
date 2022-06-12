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

{
    "seed_everything": 42,
    "dataset_batch_size": 32,
    "dataset_num_workers": 64,
    "dataset_root": "/shared/ml/datasets/vision/",
    "dataset_val_split": 0.20,
    "trainer_gpu_devices": {
        "2": 2,
    },
    "trainer_monitor_metric": "val_accuracy",
    "trainer_name": "pl-flash-cli",
    "trainer_root_dir": "model_training/lightning",
}
