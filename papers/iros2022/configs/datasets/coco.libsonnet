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

local base_dataset = import "./base_dataset.libsonnet";

local dataset_subcommand_name_var = "from_coco";

base_dataset + {
    local config = self,

    dataset_subcommand_name:: dataset_subcommand_name_var,

    dataset_name:: "coco",
    #dataset_num_classes:: 10,

    dataset_train_annotations:: "annotations/instances_train2017.json",
    dataset_train_folder:: "images/train_2017",
    dataset_test_annotations:: "annotations/instances_val2017.json",
    dataset_test_folder:: "images/val_2017",
    # while val is defined, we don't use it since we use val2017 as test dataset
    # and we use val_split to get validation images from the training images
    dataset_val_annotations:: "annotations/instances_val2017.json",
    dataset_val_folder:: "images/val_2017",

    dataset_complete_test_annotations:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_test_annotations,
    ),
    dataset_complete_test_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_test_folder,
    ),
    dataset_complete_train_annotations:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_train_annotations,
    ),
    dataset_complete_train_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_train_folder,
    ),
    dataset_complete_val_annotations:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_val_annotations,
    ),
    dataset_complete_val_dir:: utils.joinPath(
        self.dataset_complete_dir,
        self.dataset_val_folder,
    ),

    [dataset_subcommand_name_var]+: {

        # The batch size to be used by the DataLoader.
        # Defaults to 1. (type: int, default: 4)
        "batch_size": config.dataset_batch_size,

        # The size to resize images (and their bounding boxes) to.
        # (type: Tuple[int, int], default: (128, 128))
        "image_size": [128, 128],

        # The number of workers to use for parallelized loading.
        # Defaults to None which equals the number of available CPU threads,
        # or 0 for Windows or Darwin platform. (type: int, default: 0)
        "num_workers": config.dataset_num_workers,

        # The :class:`~flash.core.data.io.output_transform.OutputTransform`
        # to use when constructing the
        # :class:`~flash.core.data.data_pipeline.DataPipeline`. If ``None``, a plain
        # :class:`~flash.core.data.io.output_transform.OutputTransform` will be used.
        # (type: Optional[OutputTransform], default: null)
        "output_transform": null,

        # The folder containing the predict data. (type: Optional[str], default: null)
        "predict_folder": null,

        # The dictionary of transforms to use during predicting which maps
        # :class:`~flash.core.data.io.input_transform.InputTransform` hook names
        # to callable transforms. (type: Optional[Dict[str, Callable]], default: null)
        "predict_transform": null,

        # A sampler following the :class:`~torch.utils.data.sampler.Sampler` type.
        # Will be passed to the DataLoader for the training dataset. Defaults to None.
        # (type: Optional[Type[Sampler]], default: null)
        "sampler": null,

        # The COCO format annotation file. (type: Optional[str], default: null)
        "test_ann_file": config.dataset_complete_test_annotations,

        # The folder containing the test data. (type: Optional[str], default: null)
        "test_folder": config.dataset_complete_test_dir,

        # The dictionary of transforms to use during testing which maps
        # :class:`~flash.core.data.io.input_transform.InputTransform` hook names
        # to callable transforms. (type: Optional[Dict[str, Callable]], default: null)
        "test_transform": null,

        # The COCO format annotation file. (type: Optional[str], default: null)
        "train_ann_file": config.dataset_complete_train_annotations,

        # The folder containing the train data. (type: Optional[str], default: null)
        "train_folder": config.dataset_complete_train_dir,

        # The dictionary of transforms to use during training which maps
        # :class:`~flash.core.data.io.input_transform.InputTransform` hook names
        # to callable transforms. (type: Optional[Dict[str, Callable]], default: null)
        "train_transform": null,

        # The COCO format annotation file. (type: Optional[str], default: null)
        "val_ann_file": null,

        # The folder containing the validation data.
        # (type: Optional[str], default: null)
        "val_folder": null,

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
