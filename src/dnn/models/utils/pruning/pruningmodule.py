# this class implements the basic interface for pruning, based on ModuleABC
import abc
import collections.abc

import torch
import torch.nn.utils.prune
import pytorch_lightning as pl

from ... import moduleabc

DEFAULT_PRUNING_ARGS = {}
DEFAULT_PRUNING_FUNCTION = torch.nn.utils.prune.Identity  # mask is full of 1s


class PruningModule(moduleabc.ModuleABC):
    def __init__(self, pruning_function=DEFAULT_PRUNING_FUNCTION,
                       pruning_args=DEFAULT_PRUNING_ARGS,
                       *args, **kwargs):
        kwargs.update({'pruning_function': pruning_function,
                       'pruning_args': pruning_args})

        super().__init__(*args, **kwargs)

        self._pruning_function = pruning_function
        self._pruning_args = pruning_args

        self._pruning_enabled = False

    # we need to implement pruning for a list of modules, and for each of them
    # we need to add sparsity in another module, which must be done after
    # the pruning, and must be reset each time as for the pruning

    # convert pruning_args into module names [(module, name), ], and for each
    # of them we require the correct arguments for the pruning

    def enable_pruning(self, *args, **kwargs):
        raise NotImplementedError

    def disable_pruning(self, *args, **kwargs):
        raise NotImplementedError

    def make_pruning_permanent(self, *args, **kwargs):
        raise NotImplementedError
