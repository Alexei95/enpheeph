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

import enpheeph.injections.plugins.mask.abc.lowleveltorchmaskpluginabc
import enpheeph.utils.functions
import enpheeph.utils.imports

if typing.TYPE_CHECKING or (
    enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.NUMPY_NAME]
    and enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.TORCH_NAME]
):
    import numpy
    import torch


class NumPyPyTorchMaskPlugin(
    # we disable black to avoid too long line issue in flake8
    # fmt: off
    (
        enpheeph.injections.plugins.mask.
        lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
    ),
    # fmt: on
):
    def to_torch(self, array: "numpy.ndarray") -> "torch.Tensor":
        return torch.from_numpy(array)

    def from_torch(self, tensor: "torch.Tensor") -> "numpy.ndarray":
        return tensor.numpy()

    def to_bitwise_type(self, array: "numpy.ndarray") -> "numpy.ndarray":
        return array.view(numpy.dtype(f"u{array.dtype.itemsize}"))

    def to_target_type(
        self, array: "numpy.ndarray", target: "numpy.ndarray"
    ) -> "numpy.ndarray":
        return array.view(target.dtype)

    def make_mask_array_from_index(
        self,
        int_mask: int,
        mask_index: enpheeph.utils.typings.AnyIndexType,
        int_fill_value: int,
        shape: typing.Sequence[int],
        torch_placeholder: "torch.Tensor",
    ) -> "numpy.ndarray":
        # we convert the placeholder
        placeholder = self.from_torch(torch_placeholder)
        # we convert the integer value representing the fill value into
        # an element with unsigned type and correct size
        fill_value = numpy.array(
            int_fill_value,
            dtype=numpy.dtype(f"u{str(placeholder.dtype.itemsize)}"),
        )
        # we broadcast it onto the correct shape
        # NOTE: broadcast_to creates a view, so the view is not writeable
        # we have to make a copy of it to be able to write the mask in it
        mask = numpy.broadcast_to(fill_value, shape).copy()
        # we set the indices to the mask value
        mask[mask_index] = int_mask
        # we convert the mask to the right dtype
        mask = mask.view(dtype=placeholder.dtype)
        # we return the mask
        return mask

    def make_mask_array_from_mask(
        self,
        int_mask: int,
        mask: enpheeph.utils.typings.AnyMaskType,
        int_fill_value: int,
        shape: typing.Sequence[int],
        torch_placeholder: "torch.Tensor",
    ) -> "numpy.ndarray":
        # we convert the placeholder
        placeholder = self.from_torch(torch_placeholder)
        # we convert the integer value representing the fill value into
        # an element with unsigned type and correct size
        fill_value = numpy.array(
            int_fill_value,
            dtype=numpy.dtype(f"u{str(placeholder.dtype.itemsize)}"),
        )
        # we broadcast it onto the correct shape
        # NOTE: broadcast_to creates a view, so the view is not writeable
        # we have to make a copy of it to be able to write the mask in it
        fill_value_array = numpy.broadcast_to(fill_value, shape).copy()
        # we create an array with the same shape as the input for the int_mask
        # as then we will choose the correct element using numpy.where
        # since our mask is a boolean array
        int_mask_array: "numpy.ndarray" = (
            numpy.ones(
                shape,
                dtype=numpy.dtype(f"u{str(placeholder.dtype.itemsize)}"),
            )
            * int_mask
        )
        # we set the indices to the mask value
        # mask must become an array
        final_mask = numpy.where(numpy.asarray(mask), int_mask_array, fill_value_array)
        # we convert the mask to the right dtype
        final_mask = final_mask.view(dtype=placeholder.dtype)
        # we return the mask
        return final_mask

    def make_mask_array(
        self,
        int_mask: int,
        int_fill_value: int,
        shape: typing.Sequence[int],
        torch_placeholder: "torch.Tensor",
        mask: typing.Optional[enpheeph.utils.typings.AnyMaskType] = None,
        mask_index: typing.Optional[enpheeph.utils.typings.AnyIndexType] = None,
    ) -> "numpy.ndarray":
        if mask is None and mask_index is None:
            raise ValueError("only one between mask and mask_index can be None")
        elif mask is not None and mask_index is not None:
            raise ValueError(
                "at most one between mask and mask_index can be different from None"
            )
        elif mask is None:
            return self.make_mask_array_from_index(
                int_mask=int_mask,
                mask_index=mask_index,
                int_fill_value=int_fill_value,
                shape=shape,
                torch_placeholder=torch_placeholder,
            )
        elif mask_index is None:
            return self.make_mask_array_from_mask(
                int_mask=int_mask,
                mask=mask,
                int_fill_value=int_fill_value,
                shape=shape,
                torch_placeholder=torch_placeholder,
            )
