function(
    trainer_config,
)
trainer_config + {
    "trainer"+: {
        "callbacks" +: [
            {
                "class_path": "pytorch_lightning.callbacks.ModelPruning",
                "init_args": {
                    # (Union[int, float, Callable[[int], Union[int, float]]]) –
                    # Quantity of parameters to prune:
                    # float. Between 0.0 and 1.0.
                    # Represents the fraction of parameters to prune.
                    # int. Represents the absolute number of parameters to prune.
                    # Callable. For dynamic values. Will be called every epoch.
                    # Should return a value.
                    "amount": 0.8,

                    # (Union[bool, Callable[[int], bool]]) –
                    # Whether to apply pruning.
                    # bool. Always apply it or not.
                    # Callable[[epoch], bool]. For dynamic values.
                    # Will be called every epoch.
                    "apply_pruning": true,

                    # (bool) – Whether to remove all reparametrization pre-hooks
                    # and apply masks when training ends or the model is saved.
                    # if false it becomes impossible to save the model
                    "make_pruning_permanent": true,

                    # (Sequence[Tuple[Module, str]]) – List of tuples
                    # (nn.Module, "parameter_name_string").
                    "parameters_to_prune": [],

                    # (Optional[List[str]]) – List of parameter names
                    # to be pruned from the nn.Module.
                    # Can either be "weight" or "bias".
                    "parameter_names": null,

                    # (bool) – whether to apply pruning at the end of the
                    # training epoch. If this is False,
                    # then the check runs at the end of the validation epoch.
                    "prune_on_train_epoch_end": true,

                    # (Optional[int]) – If you are using a structured pruning
                    # method you need to specify the dimension.
                    "pruning_dim": null,

                    # (Union[Callable, str]) – Function from torch.nn.utils.prune
                    # module or your own PyTorch BasePruningMethod subclass.
                    # Can also be string e.g. “l1_unstructured”.
                    # See pytorch docs for more details.
                    "pruning_fn": "ln_unstructured",

                    # (Optional[int]) – If you are using ln_structured
                    # you need to specify the norm.
                    "pruning_norm": 2,

                    # (bool) – Used with use_lottery_ticket_hypothesis.
                    # If True, the model parameters will be resampled, otherwise,
                    # the exact original parameters will be used.
                    "resample_parameters": true,

                    # (bool) – Whether to apply pruning globally on the model.
                    # If parameters_to_prune is provided,
                    # global unstructured will be restricted on them.
                    "use_global_unstructured": true,

                    # (Union[bool, Callable[[int], bool]]) –
                    # See The lottery ticket hypothesis:
                    # bool. Whether to apply it or not.
                    # Callable[[epoch], bool]. For dynamic values.
                    # Will be called every epoch.
                    "use_lottery_ticket_hypothesis": true,

                    # (int) – Verbosity level. 0 to disable,
                    # 1 to log overall sparsity, 2 to log per-layer sparsity
                    "verbose": 2,
                },
            },
        ],
    },
}
