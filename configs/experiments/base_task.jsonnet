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

local defaults = import "../defaults.libsonnet";
local utils = import "../utils.libsonnet";

local base = import "../base.libsonnet";
local pltrainer = import "../trainers/pytorch_lightning_trainer.libsonnet";
local pruning_callback = import "../trainers/callbacks/pruning.libsonnet";
local qat_callback = import "../trainers/callbacks/quantization_aware_training.libsonnet";
local gpu_ext = import "../trainers/extensions/gpu.libsonnet";
local flashtrainer = import "../trainers/flash_trainer.libsonnet";

local select_dataset = import "../datasets/select_dataset.libsonnet";
local select_model = import "../models/select_model.libsonnet";

base +
pltrainer +
gpu_ext +
flashtrainer +

# order is not important as the final object is built at the end
select_model(std.extVar("model_task")) +
select_dataset(std.extVar("dataset_name")) +

pruning_callback(std.extVar("enable_pruning")) +
qat_callback(std.extVar("enable_qat")) +
{
    dataset_batch_size:: std.extVar("dataset_batch_size"),
    dataset_num_workers:: 64,
    dataset_root:: std.extVar("dataset_root"),

    model_backbone:: std.extVar("model_backbone"),
    model_head:: std.extVar("model_head"),

    trainer_gpu_devices:: {"2": 2},
    trainer_name:: super.trainer_name,
    trainer_root_dir:: "checkpoints/"
}
