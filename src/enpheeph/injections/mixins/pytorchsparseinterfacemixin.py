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

import enpheeph.injections.plugins.indexing.abc.indexingpluginabc
import enpheeph.injections.plugins.mask.abc.lowleveltorchmaskpluginabc
import enpheeph.injections.abc.pytorchinjectionabc
import enpheeph.utils.dataclasses
import enpheeph.utils.functions
import enpheeph.utils.imports
import enpheeph.utils.typings

if typing.TYPE_CHECKING:
    import torch
elif enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.TORCH_NAME]:
    import torch


class PyTorchSparseInterfaceMixin(abc.ABC):
    # we need the index plugin to simplify the handling of the indices
    indexing_plugin: (
        enpheeph.injections.plugins.indexing.abc.indexingpluginabc.IndexingPluginABC
    )
    # the used variables in the functions, must be initialized properly
    location: enpheeph.utils.dataclasses.BaseInjectionLocation

    def _check_sparse_index_flag(self) -> bool:
        # mypy has some issues in recognizing the enum names if taken from a name itself
        # e.g. A.a.a
        # we use separate values to avoid this issue
        # however we still require typing from the enum,
        # which limits the customizability of the interface, as before it could be any
        # compatible enum but now it must be this specific one
        # **NOTE**: a possible alternative is using .value at the end to extract the
        # correct enum, which does nothing
        # however value returns the integer value, so it is still not a clean trick
        sparse_index_flag = (
            self.location.parameter_type.Sparse | self.location.parameter_type.Index
        )
        return sparse_index_flag in self.location.parameter_type

    def _check_sparse_value_flag(self) -> bool:
        # mypy has some issues in recognizing the enum names if taken from a name itself
        # e.g. A.a.a
        # we use separate values to avoid this issue
        # however we still require typing from the enum,
        # which limits the customizability of the interface, as before it could be any
        # compatible enum but now it must be this specific one
        # **NOTE**: a possible alternative is using .value at the end to extract the
        # correct enum, which does nothing
        # however value returns the integer value, so it is still not a clean trick
        sparse_value_flag = (
            self.location.parameter_type.Sparse | self.location.parameter_type.Value
        )
        return sparse_value_flag in self.location.parameter_type

    def get_sparse_injection_parameter(
        self,
        tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        sparse_target = tensor.to_sparse()

        if self._check_sparse_index_flag():
            target = sparse_target.indices()
        elif self._check_sparse_value_flag():
            target = sparse_target.values()
        else:
            raise ValueError("This operation is not supported with sparse tensors")

        return target

    def set_sparse_injection_parameter(
        self,
        target: "torch.Tensor",
        new_value: "torch.Tensor",
    ) -> "torch.Tensor":
        sparse_target = target.to_sparse()

        if self._check_sparse_index_flag():
            other_sparse_element = sparse_target.values()
            new_target = torch.sparse_coo_tensor(
                indices=new_value, values=other_sparse_element
            )
        elif self._check_sparse_value_flag():
            other_sparse_element = sparse_target.indices()
            new_target = torch.sparse_coo_tensor(
                indices=other_sparse_element, values=new_value
            )
        else:
            raise ValueError("This operation is not supported with sparse tensors")

        return new_target
