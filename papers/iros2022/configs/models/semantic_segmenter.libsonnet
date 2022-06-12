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

    model_base_name:: "semantic_segmenter",
    model_learning_rate:: 0.001,
    # this metric is required for the trainer callbacks
    model_monitor_metric:: "val_cross_entropy",
    model_task:: "semantic_segmentation",

    # we need to remove deterministic otherwise the default loss function
    # does not run
    trainer_deterministic:: false,

    # ``SemanticSegmentation`` is a :class:`~flash.Task`
    # for semantic segmentation of images. For more details, see
    "model": {
        # A string or (model, num_features) tuple to use to compute image features,
        # defaults to ``"resnet18"``.
        # (type: Union[str, Tuple[Module, int]], default: resnet18)
        "backbone": config.model_backbone,

        #   (type: Optional[Dict], default: null)
        "backbone_kwargs"+: config.model_backbone_kwargs,

        #   (type: Union[function, Module, null], default: null)
        "head": config.model_head,

        # Learning rate to use for training, defaults to ``1e-3``.
        # (type: float, default: 0.001)
        "learning_rate": config.model_learning_rate,

        # Loss function for training, defaults to
        # :func:`torch.nn.functional.cross_entropy`.
        # (type: Union[Callable, Mapping, Sequence, null], default: null)
        "loss_fn"+: config.model_loss_fn,

        # The LR scheduler to use during training.
        # (type: Union[str, Callable, Tuple[str, Dict[str, Any]],
        # Tuple[str, Dict[str, Any], Dict[str, Any]], null], default: null)
        "lr_scheduler": null,

        # Metrics to compute for training and evaluation.
        # Can either be an metric from the `torchmetrics`
        # package, a custom metric inheriting from `torchmetrics.Metric`,
        # a callable function or a list/dict
        # containing a combination of the aforementioned.
        # In all cases, each metric needs to have the signature
        # `metric(preds,target)` and return a single scalar tensor.
        # Defaults to :class:`torchmetrics.IOU`.
        # (type: Union[Metric, Mapping, Sequence, null], default: null)
        "metrics":
            if std.length(config.model_metrics) > 0 then
                std.objectValues(config.model_metrics),

        # Whether the targets are multi-label or not. (type: bool, default: False)
        "multi_label": config.model_multi_label,

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
        # "output": null,

        # :class:`~flash.core.data.io.output_transform.OutputTransform`
        # use for post processing samples.
        # (type: Optional[OutputTransform], default: null)
        # "output_transform": null,

        # A bool or string to specify the pretrained weights of the backbone,
        # defaults to ``True``
        # which loads the default supervised pretrained weights.
        # (type: Union[bool, str], default: True)
        "pretrained": true,

        # A instance of :class:`~flash.core.data.process.Serializer`
        # or a mapping consisting of such
        # to use when serializing prediction outputs.
        # (type: Union[Serializer, Mapping[str, Serializer], null], default: null)
        # **IMPORTANT**: it does not work in Flash 0.5.2
        # "serializer": null,
    },
}
