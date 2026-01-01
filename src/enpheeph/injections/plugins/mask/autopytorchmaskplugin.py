# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2026 Alessio "Alexei95" Colucci
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

import enpheeph.injections.plugins.mask
import enpheeph.injections.plugins.mask.abc.lowleveltorchmaskpluginabc
import enpheeph.utils.functions
import enpheeph.utils.imports
import enpheeph.utils.typings

if typing.TYPE_CHECKING:
    import torch

    import enpheeph.injections.plugins.mask.numpypytorchmaskplugin
    import enpheeph.injections.plugins.mask.cupypytorchmaskplugin
else:
    if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.TORCH_NAME]:
        import torch
    if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.CUPY_NAME]:
        import enpheeph.injections.plugins.mask.cupypytorchmaskplugin
    if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.NUMPY_NAME]:
        import enpheeph.injections.plugins.mask.numpypytorchmaskplugin


class AutoPyTorchMaskPlugin(
    # we disable black to avoid too long line issue in flake8
    # fmt: off
    (
        enpheeph.injections.plugins.mask.
        lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
    ),
    # fmt: on
):
    CPU_TORCH_DEVICE = "cpu"
    GPU_TORCH_DEVICE = "cuda"

    FROM_TORCH = {
        CPU_TORCH_DEVICE: enpheeph.injections.plugins.mask.NumPyPyTorchMaskPlugin()
        if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.NUMPY_NAME]
        else None,
        GPU_TORCH_DEVICE: enpheeph.injections.plugins.mask.CuPyPyTorchMaskPlugin()
        if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.CUPY_NAME]
        else None,
    }
    TO_TORCH = {
        enpheeph.utils.imports.CUPY_NAME: FROM_TORCH[GPU_TORCH_DEVICE],
        enpheeph.utils.imports.NUMPY_NAME: FROM_TORCH[CPU_TORCH_DEVICE],
    }

    def _get_from_torch_plugin_instance(
        self, tensor: "torch.Tensor"
    ) -> (
        enpheeph.injections.plugins.mask.abc.lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
    ):
        plugin_instance = self.FROM_TORCH[tensor.device.type]
        if plugin_instance is None:
            raise ValueError(
                "Check the requirements as the current plugin is " "not available"
            )
        return plugin_instance

    def _get_to_torch_plugin_instance(
        self,
        array: enpheeph.utils.typings.ArrayType,
    ) -> (
        enpheeph.injections.plugins.mask.abc.lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
    ):
        plugin_instance = self.TO_TORCH[
            typing.cast(
                str,
                enpheeph.utils.functions.get_object_library(array),
            )
        ]
        if plugin_instance is None:
            raise ValueError(
                "Check the requirements as the current plugin is " "not available"
            )
        return plugin_instance

    def to_torch(self, array: enpheeph.utils.typings.ArrayType) -> "torch.Tensor":
        plugin_instance = self._get_to_torch_plugin_instance(array)
        return typing.cast("torch.Tensor", plugin_instance.to_torch(array))

    def from_torch(self, tensor: "torch.Tensor") -> enpheeph.utils.typings.ArrayType:
        plugin_instance = self._get_from_torch_plugin_instance(tensor)
        return plugin_instance.from_torch(tensor)

    def to_bitwise_type(
        self, array: enpheeph.utils.typings.ArrayType
    ) -> enpheeph.utils.typings.ArrayType:
        plugin_instance = self._get_to_torch_plugin_instance(array)
        return plugin_instance.to_bitwise_type(array)

    def to_target_type(
        self,
        array: enpheeph.utils.typings.ArrayType,
        target: enpheeph.utils.typings.ArrayType,
    ) -> enpheeph.utils.typings.ArrayType:
        plugin_instance = self._get_to_torch_plugin_instance(array)
        return plugin_instance.to_target_type(array, target)

    def make_mask_array(
        self,
        int_mask: int,
        int_fill_value: int,
        shape: typing.Sequence[int],
        torch_placeholder: "torch.Tensor",
        mask: typing.Optional[enpheeph.utils.typings.AnyMaskType] = None,
        mask_index: typing.Optional[enpheeph.utils.typings.AnyIndexType] = None,
    ) -> enpheeph.utils.typings.ArrayType:
        return self._get_from_torch_plugin_instance(torch_placeholder).make_mask_array(
            int_mask=int_mask,
            mask_index=mask_index,
            mask=mask,
            int_fill_value=int_fill_value,
            shape=shape,
            torch_placeholder=torch_placeholder,
        )
