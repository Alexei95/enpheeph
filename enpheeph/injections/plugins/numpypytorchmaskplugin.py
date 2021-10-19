import functools
import typing

import torch

import enpheeph.injections.plugins.lowleveltorchmaskpluginabc
import enpheeph.utils.functions

numpy = enpheeph.utils.functions.safe_import('numpy')

@enpheeph.utils.functions.test_library_access_wrapper(numpy, 'numpy')
class NumPyPyTorchMaskPlugin(
        (
                enpheeph.injections.plugins.lowleveltorchmaskpluginabc.
                LowLevelTorchMaskPluginABC
        ),
):
    def to_torch(self, array: 'numpy.ndarray') -> torch.Tensor:
        return torch.from_numpy(array)

    def from_torch(self, tensor: torch.Tensor) -> 'numpy.ndarray':
        return tensor.numpy()

    def to_bitwise_type(self, array: 'numpy.ndarray') -> 'numpy.ndarray':
        return array.view(numpy.dtype(f'u{array.dtype.itemsize}'))

    def to_target_type(
            self,
            array: 'numpy.ndarray',
            target: 'numpy.ndarray'
    ) -> 'numpy.ndarray':
        return array.view(target.dtype)

    def make_mask_array(
            self,
            int_mask: int,
            mask_index: enpheeph.utils.typings.IndexType,
            int_fill_value: int,
            shape: typing.Sequence[int],
            torch_placeholder: torch.Tensor
    ) -> 'numpy.ndarray':
        # we convert the placeholder
        placeholder = self.from_torch(torch_placeholder)
        # we convert the integer value representing the fill value into
        # an element with unsigned type and correct size
        fill_value = numpy.array(
                int_fill_value,
                dtype=numpy.dtype(f'u{str(placeholder.dtype.itemsize)}')
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