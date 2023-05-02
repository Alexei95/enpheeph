# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2023 Alessio "Alexei95" Colucci
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

import typing

import enpheeph.injections.abc.faultabc
import enpheeph.injections.abc.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmaskmixin
import enpheeph.injections.mixins.pytorchtensorobjectvalidatormixin
import enpheeph.injections.plugins.mask.abc.lowleveltorchmaskpluginabc
import enpheeph.utils.dataclasses

# we move this import down
if typing.TYPE_CHECKING:
    import torch


class OutputPyTorchFault(
    enpheeph.injections.abc.faultabc.FaultABC,
    enpheeph.injections.abc.pytorchinjectionabc.PyTorchInjectionABC,
    enpheeph.injections.mixins.pytorchmaskmixin.PyTorchMaskMixin,
    (
        # fmt: off
        enpheeph.injections.mixins.
        pytorchtensorobjectvalidatormixin.PyTorchTensorObjectValidatorMixin
        # fmt: on
    ),
):
    handle: typing.Optional["torch.utils.hooks.RemovableHandle"]
    # we need the index plugin to simplify the handling of the indices
    indexing_plugin: (
        enpheeph.injections.plugins.indexing.abc.indexingpluginabc.IndexingPluginABC
    )
    location: enpheeph.utils.dataclasses.FaultLocation
    low_level_plugin: (
        # black has issues with long names
        # fmt: off
        enpheeph.injections.plugins.mask.
        lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
        # fmt: on
    )
    mask: typing.Optional["torch.Tensor"]

    def __init__(
        self,
        indexing_plugin: (
            enpheeph.injections.plugins.indexing.abc.indexingpluginabc.IndexingPluginABC
        ),
        location: enpheeph.utils.dataclasses.FaultLocation,
        low_level_torch_plugin: (
            # black has issues with long names
            # fmt: off
            enpheeph.injections.plugins.mask.
            lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
            # fmt: on
        ),
    ) -> None:
        super().__init__()

        self.indexing_plugin = indexing_plugin
        self.location = location
        self.low_level_plugin = low_level_torch_plugin

        self.handle = None
        self.mask = None

    @property
    def module_name(self) -> str:
        return self.location.module_name

    def output_fault_hook(
        self,
        module: "torch.nn.Module",
        input: typing.Union[typing.Tuple["torch.Tensor"], "torch.Tensor"],
        output: "torch.Tensor",
    ) -> "torch.Tensor":
        self.generate_mask(output, tensor_only=True)

        masked_output = self.inject_mask(output, tensor_only=False)

        return masked_output

    def setup(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        self.handle = module.register_forward_hook(self.output_fault_hook)

        return module
