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
import enpheeph.utils.data_classes
import enpheeph.utils.functions
import enpheeph.utils.imports
import enpheeph.utils.typings

if typing.TYPE_CHECKING:
    import torch


class PyTorchSparseInterfaceMixin(abc.ABC):
    # we need the index plugin to simplify the handling of the indices
    indexing_plugin: (
        enpheeph.injections.plugins.indexing.abc.indexingpluginabc.IndexingPluginABC
    )
    # the used variables in the functions, must be initialized properly
    location: enpheeph.utils.data_classes.BaseInjectionLocation

    def get_sparse_injection_parameter(
        self,
        tensor: "torch.Tensor",
    ) -> "torch.Tensor":
        sparse_output = tensor.to_sparse()

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
        sparse_value_flag = (
            self.location.parameter_type.Sparse | self.location.parameter_type.Value
        )
        if sparse_index_flag in self.location.parameter_type:
            target = sparse_output.indices()
        elif sparse_value_flag in self.location.parameter_type:
            target = sparse_output.values()
        else:
            raise ValueError("This operation is not supported with sparse tensors")

        return target

    def set_sparse_injection_parameter(
        self,
        target: "torch.Tensor",
        new_value: "torch.Tensor",
    ) -> "torch.Tensor":
        import torch

        sparse_output = target.to_sparse()

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
        sparse_value_flag = (
            self.location.parameter_type.Sparse | self.location.parameter_type.Value
        )
        if sparse_index_flag in self.location.parameter_type:
            other_target_value = sparse_output.values()
            new_target = torch.sparse_coo_tensor(
                indices=new_value, values=other_target_value
            )
        elif sparse_value_flag in self.location.parameter_type:
            other_target_value = sparse_output.indices()
            new_target = torch.sparse_coo_tensor(
                indices=other_target_value, values=new_value
            )
        else:
            raise ValueError("This operation is not supported with sparse tensors")

        return new_target
