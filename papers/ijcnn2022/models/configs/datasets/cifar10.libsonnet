# we define the config as a function,
# so that we require the arguments to be passed to work properly
function(
    batch_size,
    dataset_dir,
    num_workers,
    val_split,
)
{
    num_classes:: 10,

    "from_datasets": {

        "batch_size": batch_size,

        "num_workers": num_workers,

        "test_dataset": {
            "class_path": "torchvision.datasets.CIFAR10",
            "init_args": {
                "download": true,

                "root": dataset_dir,

                "train": false,
            },
        },

        "train_dataset": {
            "class_path": "torchvision.datasets.CIFAR10",
            "init_args": {
                "download": true,

                "root": dataset_dir,

                "train": true,
            },
        },

        "val_split": val_split,
    },
}
