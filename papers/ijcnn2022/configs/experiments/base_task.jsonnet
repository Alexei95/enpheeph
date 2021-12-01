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
