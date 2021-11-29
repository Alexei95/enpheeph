local base_config = import "../base.libsonnet";

local pltrainer_config = import "../trainers/pytorch_lightning_trainer.libsonnet";
local flashtrainer_config = import "../trainers/flash_trainer.libsonnet";
local gpu_ext_config = import "../trainers/extensions/gpu.libsonnet";
local pruning_callback_config = import "../trainers/callbacks/pruning.libsonnet";
local qat_callback_config = import "../trainers/callbacks/quantization_aware_training.libsonnet";

local dataset_config = import "../datasets/cifar10.libsonnet";

local model_config = import "../models/image_classifier.libsonnet";

local defaults = import "../defaults.libsonnet";
local utils = import "../utils.libsonnet";

local backbone_local = "resnet18";
local batch_size_local = 32;
local name_local = "image_classification_cifar10_resnet152";

# similarly to the base config we use a function with defaults so that it
# can be further imported but it has default parameters
function(
    backbone=backbone_local,
    batch_size=batch_size_local,
    dataset_root_dir=defaults.dataset_root_dir,
    gpu_devices=defaults.gpu_devices,
    name=name_local,
    num_workers=defaults.num_workers,
    root_dir=defaults.root_dir,
    seed_everything=defaults.seed_everything,
    val_split=defaults.val_split,
)
local dataset = dataset_config(
    batch_size=batch_size,
    dataset_dir=dataset_root_dir + "/" + "CIFAR10",
    num_workers=num_workers,
    val_split=val_split,
);
local model = model_config(
    backbone=backbone,
    num_classes=dataset.num_classes,
);
local base = base_config(
    seed_everything=seed_everything,
);
# we need to use a local to override the nested fields
# as the config needs to be passed to be properly overridden
local base_trainer = pltrainer_config(
    monitor_metric=model.monitor_metric,
    name=name,
    root_dir=root_dir,
);
local gpu_trainer = gpu_ext_config(
    devices=gpu_devices,
    trainer_config=base_trainer,
);
local callback_trainer_1 = pruning_callback_config(
    trainer_config=gpu_trainer,
);
local callback_trainer_2 = qat_callback_config(
    trainer_config=callback_trainer_1,
);
local trainer = flashtrainer_config(
    base_trainer_config=callback_trainer_2,
);

# we can sum as the right-most dict overwrites the left-most one
base + trainer + model + dataset
