# -*- coding: utf-8 -*-
import typing

import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
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

    def make_mask_array(
        self,
        int_mask: int,
        mask_index: enpheeph.utils.typings.IndexMultiDType,
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
