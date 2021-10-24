# -*- coding: utf-8 -*-
import abc
import sys
import typing

import torch
import torch.utils.hooks

import enpheeph.injections.plugins.lowleveltorchmaskpluginabc
import enpheeph.injections.pytorchinjectionabc
import enpheeph.utils.data_classes
import enpheeph.utils.typings


class PyTorchMaskMixIn(abc.ABC):
    # mask is both set in self and returned
    def register_forward_hook(
        self,
        module: torch.nn.Module,
        fn: typing.Callable[
            [torch.nn.Module, torch.Tensor, torch.Tensor],
            typing.Optional[torch.Tensor],
        ],
    ) -> torch.utils.hooks.RemovableHandle:
        return module.register
