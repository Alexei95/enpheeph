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

function(enable_pruning=true)
    if enable_pruning then
        {
            local config = self,

            pruning_enabled:: true,

            pruning_amount:: 0.8,
            pruning_at_train_epoch_end:: true,
            pruning_function:: "l1_unstructured",
            pruning_norm:: null,
            pruning_names+:: {},
            pruning_parameters+:: {},
            pruning_verbose:: 2,

            trainer_callbacks+:: {
                "pruning"+: {
                    "class_path": "pytorch_lightning.callbacks.ModelPruning",
                    "init_args"+: {
                        # (Union[int, float, Callable[[int], Union[int, float]]]) –
                        # Quantity of parameters to prune:
                        # float. Between 0.0 and 1.0.
                        # Represents the fraction of parameters to prune.
                        # int. Represents the absolute number of parameters to prune.
                        # Callable. For dynamic values. Will be called every epoch.
                        # Should return a value.
                        # **IMPORTANT**: there is still the same problem
                        # with the callables
                        # so signatures with callables cannot be passed as it explodes
                        # "amount": config.pruning_amount,

                        # (Union[bool, Callable[[int], bool]]) –
                        # Whether to apply pruning.
                        # bool. Always apply it or not.
                        # Callable[[epoch], bool]. For dynamic values.
                        # Will be called every epoch.
                        # **IMPORTANT**: there is still the same problem
                        # with the callables
                        # so signatures with callables cannot be passed as it explodes
                        # "apply_pruning": true,

                        # (bool) – Whether to remove all reparametrization pre-hooks
                        # and apply masks when training ends or the model is saved.
                        # if false it becomes impossible to save the model
                        "make_pruning_permanent": true,

                        # (Sequence[Tuple[Module, str]]) – List of tuples
                        # (nn.Module, "parameter_name_string").
                        "parameters_to_prune"+: std.objectValues(config.pruning_parameters),

                        # (Optional[List[str]]) – List of parameter names
                        # to be pruned from the nn.Module.
                        # Can either be "weight" or "bias".
                        "parameter_names":
                            if std.length(config.pruning_names) > 0 then
                                std.objectValues(config.pruning_names),

                        # (bool) – whether to apply pruning at the end of the
                        # training epoch. If this is False,
                        # then the check runs at the end of the validation epoch.
                        "prune_on_train_epoch_end": config.pruning_at_train_epoch_end,

                        # (Optional[int]) – If you are using a structured pruning
                        # method you need to specify the dimension.
                        "pruning_dim": null,

                        # (Union[Callable, str]) – Function from torch.nn.utils.prune
                        # module or your own PyTorch BasePruningMethod subclass.
                        # Can also be string e.g. “l1_unstructured”.
                        # See pytorch docs for more details.
                        "pruning_fn": config.pruning_function,

                        # (Optional[int]) – If you are using ln_structured
                        # you need to specify the norm.
                        "pruning_norm": config.pruning_norm,

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
                        # **IMPORTANT**: there is still the same
                        # problem with the callables
                        # so signatures with callables cannot be passed as it explodes
                        # "use_lottery_ticket_hypothesis": true,

                        # (int) – Verbosity level. 0 to disable,
                        # 1 to log overall sparsity, 2 to log per-layer sparsity
                        "verbose": config.pruning_verbose,
                    },
                },
            },
        }
    else
        {pruning_enabled:: false}
