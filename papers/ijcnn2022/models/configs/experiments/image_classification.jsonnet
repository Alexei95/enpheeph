local pltrainer_config = import "../trainers/pytorch_lightning_trainer.libsonnet";
local flashtrainer_config = import "../trainers/flash_trainer.libsonnet";

local dataset_config = import "../datasets/cifar10.libsonnet";

local model_config = import "../models/image_classifier.libsonnet";

local defaults = import "../defaults.libsonnet";

local backbone_local = "resnet18";
local batch_size_local = 32;
local name_local = "image_classification_cifar10_resnet152";
local num_classes_local = dataset_config("", "", "", "").num_classes;

# similarly to the base config we use a function with defaults so that it
# can be further imported but it has default parameters
function(
    backbone=backbone_local,
    batch_size=batch_size_local,
    dataset_root_dir=defaults.dataset_root_dir,
    name=name_local,
    num_classes=num_classes_local,
    num_workers=defaults.num_workers,
    root_dir=defaults.root_dir,
    seed_everything=defaults.seed_everything,
    val_split=defaults.val_split,
)
local dataset = dataset_config(batch_size, dataset_root_dir + "/CIFAR10/", num_workers, val_split);
local model = model_config(backbone, num_classes);
# we need to use a local to override the nested fields
# as the config needs to be passed to be properly overridden
local base_trainer = pltrainer_config(name, root_dir, seed_everything);
local trainer = flashtrainer_config(base_trainer);
# we can sum as the right-most dict overwrites the left-most one
trainer + model + dataset
