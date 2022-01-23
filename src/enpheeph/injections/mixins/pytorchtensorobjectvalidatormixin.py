# -*- coding: utf-8 -*-
import abc
import typing

import enpheeph.injections.pytorchinjectionabc
import enpheeph.utils.data_classes
import enpheeph.utils.functions
import enpheeph.utils.imports
import enpheeph.utils.typings

if (
    typing.TYPE_CHECKING
    or enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.TORCH_NAME]
):
    import torch


class PyTorchTensorObjectValidatorMixin(abc.ABC):
    @staticmethod
    def convert_tensor_to_proper_class(
        source: "torch.Tensor", target: "torch.Tensor"
    ) -> "torch.Tensor":
        # to avoid issues if we are using sub-classes like torch.nn.Parameter,
        # we call tensor.__class__ to create a new object with the proper content
        # however this cannot be done for torch.Tensor itself as it would requiring
        # copying the tensor parameter
        if target.__class__ == torch.Tensor:
            return source
        elif isinstance(source, torch.Tensor):
            return target.__class__(source)
        else:
            raise TypeError("Wrong type for source")
