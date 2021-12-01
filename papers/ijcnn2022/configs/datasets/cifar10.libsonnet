local defaults = import "../defaults.libsonnet";
local utils = import "../utils.libsonnet";

local base_dataset = import "./base_dataset.libsonnet";

local dataset_subcommand_name_var = "from_datasets";

base_dataset + {
    local config = self,

    dataset_subcommand_name:: dataset_subcommand_name_var,

    dataset_name:: "cifar10",
    dataset_num_classes:: 10,

    [dataset_subcommand_name_var]+: {

        "batch_size": config.dataset_batch_size,

        "num_workers": config.dataset_num_workers,

        "test_dataset"+: {
            "class_path": "torchvision.datasets.CIFAR10",
            "init_args"+: {
                "download": true,

                "root": config.dataset_complete_dir,

                "train": false,
            },
        },

        "train_dataset"+: self.test_dataset + {
            "init_args"+: {
                "train": true,
            },
        },

        "val_split": config.dataset_val_split,
    },

    # this is required to have flash cli parse the correct config
    "subcommand": self.dataset_subcommand_name,
}
