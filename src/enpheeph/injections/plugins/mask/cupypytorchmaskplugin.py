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

import typing

import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.utils.functions
import enpheeph.utils.imports

if typing.TYPE_CHECKING or (
    enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.CUPY_NAME]
    and enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.TORCH_NAME]
):
    import cupy
    import torch
    import torch.utils.dlpack


class CuPyPyTorchMaskPlugin(
    # we disable black to avoid too long line issue in flake8
    # fmt: off
    (
        enpheeph.injections.plugins.mask.
        lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
    ),
    # fmt: on
):
    def to_torch(self, array: "cupy.ndarray") -> "torch.Tensor":
        return torch.utils.dlpack.from_dlpack(array.toDlpack())

    def from_torch(self, tensor: "torch.Tensor") -> "cupy.ndarray":
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))

    def to_bitwise_type(self, array: "cupy.ndarray") -> "cupy.ndarray":
        return array.view(cupy.dtype(f"u{array.dtype.itemsize}"))

    def to_target_type(
        self, array: "cupy.ndarray", target: "cupy.ndarray"
    ) -> "cupy.ndarray":
        return array.view(target.dtype)

    def make_mask_array_from_index(
        self,
        int_mask: int,
        mask_index: enpheeph.utils.typings.AnyIndexType,
        int_fill_value: int,
        shape: typing.Sequence[int],
        torch_placeholder: "torch.Tensor",
    ) -> "cupy.ndarray":
        # we convert the placeholder
        placeholder = self.from_torch(torch_placeholder)
        # we convert the integer value representing the fill value into
        # an element with unsigned type and correct size, as well as correct
        # device for cupy
        with placeholder.device:
            fill_value = cupy.array(
                int_fill_value,
                dtype=cupy.dtype(f"u{str(placeholder.dtype.itemsize)}"),
            )
            # we broadcast it onto the correct shape
            # we need to copy it to avoid issues with broadcasting
            mask = cupy.broadcast_to(fill_value, shape).copy()
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
    ) -> "cupy.ndarray":
        # we convert the placeholder
        placeholder = self.from_torch(torch_placeholder)
        # we convert the integer value representing the fill value into
        # an element with unsigned type and correct size, as well as correct
        # device for cupy
        with placeholder.device:
            fill_value = cupy.array(
                int_fill_value,
                dtype=cupy.dtype(f"u{str(placeholder.dtype.itemsize)}"),
            )
            # we broadcast it onto the correct shape
            # we need to copy it to avoid issues with broadcasting
            fill_value_array = cupy.broadcast_to(fill_value, shape).copy()
            # we create an array with the same shape as the input for the int_mask
            # as then we will choose the correct element using cupy.where
            # since our mask is a boolean array
            int_mask_array: "cupy.ndarray" = (
                cupy.ones(
                    shape,
                    dtype=cupy.dtype(f"u{str(placeholder.dtype.itemsize)}"),
                )
                * int_mask
            )
            # we set the indices to the mask value
            # mask must become an array
            final_mask = cupy.where(
                cupy.asarray(mask), int_mask_array, fill_value_array
            )
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
    ) -> "cupy.ndarray":
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
