import functools

try:
    import numpy
except ImportError:
    numpy = None
import torch

import enpheeph.injections.plugins.lowleveltorchmaskpluginabc


def _test_numpy_access_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if numpy is None:
            raise RuntimeError(
                    "numpy cannot be imported, "
                    "please check the installation to use this plugin"
            )
        return fn(*args, **kwargs)
    return wrapper


class NumpyTorchMaskPlugin(
        enpheeph.injections.plugins.lowleveltorchmaskpluginabc.
        LowLevelTorchMaskPluginABC,
):
    @_test_numpy_access_wrapper
    def to_torch(self, array: 'numpy.ndarray') -> torch.Tensor:
        self._test_numpy_access()
        
        return torch.from_numpy(array)

    @_test_numpy_access_wrapper
    def from_torch(self, tensor: torch.Tensor) -> 'numpy.ndarray':
        return tensor.numpy()

    @_test_numpy_access_wrapper
    def make_mask_array(
            self,
            int_mask: int,
            mask_index: enpheeph.utils.typings.TensorIndex,
            int_fill_value: int,
            shape: typing.Sequence[int, ...],
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
        mask = numpy.broadcast_to(fill_value, shape)
        # we set the indices to the mask value
        mask[mask_index] = int_mask
        # we convert the mask to the right dtype
        mask = mask.view(dtype=placeholder.dtype)
        # we return the mask
        return mask