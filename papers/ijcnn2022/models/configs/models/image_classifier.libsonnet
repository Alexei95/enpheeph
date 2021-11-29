function(
    backbone,
    num_classes,
){
    # this metric is required for the trainer callbacks
    monitor_metric:: "val_cross_entropy",

    # The ``ImageClassifier`` is a :class:`~flash.Task`
    # for classifying images. For more details, see
    "model": {
        # A string or (model, num_features) tuple to use to compute image features,
        # defaults to ``"resnet18"``.
        # (type: Union[str, Tuple[Module, int]], default: resnet18)
        "backbone": backbone,

        #   (type: Optional[Dict], default: null)
        "backbone_kwargs": null,

        #   (type: Union[function, Module, null], default: null)
        "head": null,

        # Learning rate to use for training, defaults to ``1e-3``.
        # (type: float, default: 0.001)
        "learning_rate": 0.001,

        # Loss function for training, defaults to
        # :func:`torch.nn.functional.cross_entropy`.
        # (type: Union[Callable, Mapping, Sequence, null], default: null)
        "loss_fn": null,

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
        # Defaults to :class:`torchmetrics.Accuracy`.
        # (type: Union[Metric, Mapping, Sequence, null], default: null)
        "metrics": null,

        # Whether the targets are multi-label or not. (type: bool, default: False)
        "multi_label": false,

        # Number of classes to classify. (type: Optional[int], default: null)
        "num_classes": num_classes,

        # Optimizer to use for training.
        # (type: Union[str, Callable, Tuple[str, Dict[str, Any]]], default: Adam)
        "optimizer": "Adam",

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

        # string indicating the training strategy.
        # Adjust if you want to use `learn2learn`
        # for doing meta-learning research (type: Optional[str], default: default)
        "training_strategy": "default",

        # Additional kwargs for setting the training strategy
        # (type: Optional[Dict[str, Any]], default: null)
        "training_strategy_kwargs": null,
    },
}
