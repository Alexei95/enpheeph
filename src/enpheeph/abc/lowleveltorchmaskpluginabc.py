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

import abc
import typing

import enpheeph.utils.enums
import enpheeph.utils.typings

# while 3rd party library should be before custom libraries, we move it down to avoid
# flake8 complaining, since it is a conditional import
if typing.TYPE_CHECKING:
    import torch


class LowLevelTorchMaskPluginABC(abc.ABC):
    @abc.abstractmethod
    def to_torch(
        self, array: enpheeph.utils.typings.LowLevelMaskArrayType
    ) -> "torch.Tensor":
        pass

    @abc.abstractmethod
    def from_torch(
        self, tensor: "torch.Tensor"
    ) -> enpheeph.utils.typings.LowLevelMaskArrayType:
        pass

    @abc.abstractmethod
    def to_bitwise_type(
        self, array: enpheeph.utils.typings.LowLevelMaskArrayType
    ) -> enpheeph.utils.typings.LowLevelMaskArrayType:
        pass

    @abc.abstractmethod
    def to_target_type(
        self,
        array: enpheeph.utils.typings.LowLevelMaskArrayType,
        target: enpheeph.utils.typings.LowLevelMaskArrayType,
    ) -> enpheeph.utils.typings.LowLevelMaskArrayType:
        pass

    @abc.abstractmethod
    def make_mask_array(
        self,
        int_mask: int,
        # this fill value is already final, as is the int mask
        int_fill_value: int,
        shape: typing.Sequence[int],
        torch_placeholder: "torch.Tensor",
        mask: typing.Optional[enpheeph.utils.typings.AnyMaskType] = None,
        mask_index: typing.Optional[enpheeph.utils.typings.AnyIndexType] = None,
    ) -> enpheeph.utils.typings.LowLevelMaskArrayType:
        pass
