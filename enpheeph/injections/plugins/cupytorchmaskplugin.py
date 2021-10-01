import functools

try:
    import cupy
except ImportError:
    cupy = None
import torch

import enpheeph.injections.plugins.lowleveltorchmaskpluginabc


def _test_cupy_access_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if cupy is None:
            raise RuntimeError(
                    "cupy cannot be imported, "
                    "please check the installation to use this plugin"
            )
        return fn(*args, **kwargs)
    return wrapper


class CupyTorchMaskPlugin(
        enpheeph.injections.plugins.lowleveltorchmaskpluginabc.
        LowLevelTorchMaskPluginABC,
):
    @_test_cupy_access_wrapper
    def to_torch(self, array: 'cupy.ndarray') -> torch.Tensor:
        self._test_cupy_access()
        
        return torch.utils.dlpack.from_dlpack(array.toDlpack())

    @_test_cupy_access_wrapper
    def from_torch(self, tensor: torch.Tensor) -> 'cupy.ndarray':
        return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))

    @_test_cupy_access_wrapper
    def make_mask_array(
            self,
            int_mask: int,
            mask_index: enpheeph.utils.typings.TensorIndex,
            int_fill_value: int,
            shape: typing.Sequence[int, ...],
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