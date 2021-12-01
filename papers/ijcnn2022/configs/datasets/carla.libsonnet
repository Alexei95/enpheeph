local defaults = import "../defaults.libsonnet";
local utils = import "../utils.libsonnet";

local base_dataset = import "./base_dataset.libsonnet";

local dataset_subcommand_name_var = "from_folders";

base_dataset + {
    local config = self,

    dataset_subcommand_name::dataset_subcommand_name_var,

    dataset_name:: "carla",
    dataset_num_classes:: 101,

    dataset_full_name::"carla-20180528-100vehicles-100pedestrians",

    dataset_complete_dir:: utils.joinPath(
        self.dataset_root,
        "carla-data-capture/" + std.join(
            "-",
            std.split(
                self.dataset_full_name,
                "-"
            )[1:],
        ),
    ),

    dataset_train_folder:: "CameraRGB/",
    dataset_train_target_folder:: "CameraSeg/",
    # we only use train for both train and val, test is not used at all
    dataset_test_folder:: "CameraRGB/",
    dataset_test_target_folder:: "CameraSeg/",
    dataset_val_folder:: "CameraRGB/",
    dataset_val_target_folder:: "CameraSeg/",

    dataset_complete_test_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_test_folder,
    ),
    dataset_complete_test_target_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_test_target_folder,
    ),
    dataset_complete_train_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_train_folder,
    ),
    dataset_complete_train_target_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_train_target_folder,
    ),
    dataset_complete_val_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_val_folder,
    ),
    dataset_complete_val_target_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_val_target_folder,
    ),

    [dataset_subcommand_name_var]+: {

        # The batch size to be used by the DataLoader.
        # Defaults to 1. (type: int, default: 4)
        "batch_size": config.dataset_batch_size,

        # The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
        # :class:`~flash.core.data.data_module.DataModule`.
        # (type: Optional[BaseDataFetcher], default: null)
        "data_fetcher": null,

        # The size to resize images (and their bounding boxes) to.
        # (type: Tuple[int, int], default: (128, 128))
        "image_size": [256, 256],

        # The :class:`~flash.core.data.data.InputTransform` to pass to the
        # :class:`~flash.core.data.data_module.DataModule`.
        # If ``None``, ``cls.input_transform_cls``
        # will be constructed and used. (type: Optional[InputTransform], default: null)
        "input_transform": null,

        # Mapping between a class_id and its corresponding color.
        # (type: Optional[Dict[int, Tuple[int, int, int]]], default: null)
        "labels_map": null,

        # Number of classes within the segmentation mask.
        # (type: Optional[int], default: null)
        "num_classes": config.dataset_num_classes,

        # The number of workers to use for parallelized loading.
        # Defaults to None which equals the number of available CPU threads,
        # or 0 for Windows or Darwin platform. (type: int, default: 0)
        "num_workers": config.dataset_num_workers,

        # The folder containing the predict data. (type: Optional[str], default: null)
        "predict_folder": null,

        # The dictionary of transforms to use during predicting which maps
        # :class:`~flash.core.data.io.input_transform.InputTransform` hook names
        # to callable transforms. (type: Optional[Dict[str, Callable]], default: null)
        "predict_transform": null,

        # The folder containing the test data. (type: Optional[str], default: null)
        "test_folder": null,

        # The folder containing the test targets
        # (targets must have the same file name as their
        # corresponding inputs).
        # (type: Optional[str], default: null)
        "test_target_folder": null,

        # The dictionary of transforms to use during testing which maps
        # :class:`~flash.core.data.io.input_transform.InputTransform` hook names
        # to callable transforms. (type: Optional[Dict[str, Callable]], default: null)
        "test_transform": null,

        # The folder containing the train data. (type: Optional[str], default: null)
        "train_folder": config.dataset_complete_train_dir,

        # The folder containing the train targets
        # (targets must have the same file name as their
        # corresponding inputs).
        # (type: Optional[str], default: null)
        "train_target_folder": config.dataset_complete_train_target_dir,

        # The dictionary of transforms to use during training which maps
        # :class:`~flash.core.data.io.input_transform.InputTransform` hook names
        # to callable transforms. (type: Optional[Dict[str, Callable]], default: null)
        "train_transform": null,

        # The folder containing the validation data.
        # (type: Optional[str], default: null)
        "val_folder": null,

        # The folder containing the validation targets
        # (targets must have the same file name as their
        # corresponding inputs).
        # (type: Optional[str], default: null)
        "val_target_folder": null,

        # An optional float which gives the relative amount of the training
        # dataset to use for the validation
        # dataset. (type: Optional[float], default: null)
        "val_split": config.dataset_val_split,

        # The dictionary of transforms to use during validation which maps
        # :class:`~flash.core.data.io.input_transform.InputTransform` hook names
        # to callable transforms. (type: Optional[Dict[str, Callable]], default: null)
        "val_transform": null,
    },

    # this is required to have flash cli parse the correct config
    "subcommand": self.dataset_subcommand_name,
}
