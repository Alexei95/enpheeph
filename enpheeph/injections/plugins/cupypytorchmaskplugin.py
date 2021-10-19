import functools
import typing

import torch
import torch.utils.dlpack

import enpheeph.injections.plugins.lowleveltorchmaskpluginabc
import enpheeph.utils.functions

cupy = enpheeph.utils.functions.safe_import('cupy')

@enpheeph.utils.functions.test_library_access_wrapper(cupy, 'cupy')
class CuPyPyTorchMaskPlugin(
        (
                enpheeph.injections.plugins.lowleveltorchmaskpluginabc.
                LowLevelTorchMaskPluginABC
        ),
):
    def to_torch(self, array: 'cupy.ndarray') -> torch.Tensor:
        return torch.utils.dlpack.from_dlpack(array.toDlpack())

    def from_torch(self, tensor: torch.Tensor) -> 'cupy.ndarray':
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))

    def to_bitwise_type(self, array: 'cupy.ndarray') -> 'cupy.ndarray':
        return array.view(cupy.dtype(f'u{array.dtype.itemsize}'))

    def to_target_type(
            self,
            array: 'cupy.ndarray',
            target: 'cupy.ndarray'
    ) -> 'cupy.ndarray':
        return array.view(target.dtype)

    def make_mask_array(
            self,
            int_mask: int,
            mask_index: enpheeph.utils.typings.IndexType,
            int_fill_value: int,
            shape: typing.Sequence[int],
            torch_placeholder: torch.Tensor
    ) -> 'cupy.ndarray':
        # we convert the placeholder
        placeholder = self.from_torch(torch_placeholder)
        # we convert the integer value representing the fill value into
        # an element with unsigned type and correct size, as well as correct
        # device for cupy
        with placeholder.device:
            fill_value = cupy.array(
                    int_fill_value,
                    dtype=cupy.dtype(f'u{str(placeholder.dtype.itemsize)}')
            )
        # we broadcast it onto the correct shape
        mask = cupy.broadcast_to(fill_value, shape)
        # we set the indices to the mask value
        mask[mask_index] = int_mask
        # we convert the mask to the right dtype
        mask = mask.view(dtype=placeholder.dtype)
        # we return the mask
        return mask