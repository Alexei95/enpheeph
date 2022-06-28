# -*- coding: utf-8 -*-
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

import abc
import typing

import enpheeph.injections.abc.pytorchinjectionabc
import enpheeph.utils.dataclasses
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
