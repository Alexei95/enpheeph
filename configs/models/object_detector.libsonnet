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

local base_model = import "./base_model.libsonnet";


base_model + {
    local config = self,
    model_base_name:: "object_detector",
    model_learning_rate:: 0.005,
    # this metric is required for the trainer callbacks
    model_monitor_metric:: "Precision (IoU=0.50:0.95,area=all)",
    model_task:: "object_detection",

    # The ``ObjectDetector`` is a :class:`~flash.Task` for detecting objects in images.
    # For more details, see
    "model": {
        # String indicating the backbone CNN architecture to use.
        # (type: Optional[str], default: resnet18_fpn)
        "backbone": config.model_backbone,

        # String indicating the head module to use ontop of the backbone.
        # (type: Optional[str], default: retinanet)
        "head": config.model_head,

        # The learning rate to use for training. (type: float, default: 0.005)
        "learning_rate": config.model_learning_rate,

        # The LR scheduler to use during training.
        # (type: Union[str, Callable, Tuple[str, Dict[str, Any]],
        # Tuple[str, Dict[str, Any], Dict[str, Any]], null], default: null)
        "lr_scheduler": null,

        # Number of classes to classify. (type: Optional[int], default: null)
        "num_classes":
            if !(config.model_num_classes == null) then
                config.model_num_classes
            else if std.objectHasAll(config, "dataset_num_classes") then
                # HasAll covers also hidden fields
                config.dataset_num_classes
            else
                config.model_num_classes,

        # Optimizer to use for training.
        # (type: Union[str, Callable, Tuple[str, Dict[str, Any]]], default: Adam)
        "optimizer"+: config.model_optimizer,

        # The :class:`~flash.core.data.io.output.Output`
        # to use when formatting prediction outputs.
        # (type: Optional[Output], default: null)
        "output": null,

        # dictionary containing parameters that will be used
        # during the prediction phase.
        # (type: Optional[Dict], default: null)
        "predict_kwargs": null,

        # A bool or string to specify the pretrained weights of the backbone,
        # defaults to ``True``
        # which loads the default supervised pretrained weights.
        # (type: Union[bool, str], default: True)
        "pretrained": true,
    },
}
