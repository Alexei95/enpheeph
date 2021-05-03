import copy

# PyTorch depends on Numpy, so we don't need this check, but we leave it
# to give an example on how to handle cross-library dependencies
try:
    import numpy
except ImportError:
    numpy = None
import torch


class PyTorchConverter(object):
    PYTORCH_DTYPE_TO_BITWIDTH = {
            torch.float32: 32,
            torch.float: 32,
            torch.float64: 64,
            torch.double: 64,
            torch.float16: 16,
            torch.half: 16,
            torch.bfloat16: 16,
            torch.complex32: 32,
            torch.complex64: 64,
            torch.complex128: 128,
            torch.cdouble: 128,
            torch.uint8: 8,
            torch.int8: 8,
            torch.int16: 16,
            torch.short: 16,
            torch.int32: 32,
            torch.int: 32,
            torch.int64: 64,
            torch.long: 64,
            torch.bool: 1,
    }

    @classmethod
    def single_pytorch_to_numpy(cls, element: torch.Tensor) -> 'numpy.ndarray':
        # we check we have only 1 element, with empty shape
        if element.nelement() != 1:
            raise ValueError('There must be only 1 element in the array')

        return element.squeeze().cpu().numpy()

    @classmethod
    def pytorch_to_numpy(cls, element: torch.Tensor) -> 'numpy.ndarray':
        if numpy is None:
            raise NotImplementedError(
                    'numpy is not available, '
                    'function not supported'
            )
        return element.squeeze().cpu().numpy()

    # we convert from numpy to torch tensor, with optional device and type
    @classmethod
    def numpy_to_pytorch(
            cls,
            element: 'numpy.ndarray',
            *,
            dtype: torch.dtype = None,
            device: torch.device = None,
    ) -> torch.Tensor:
        return torch.from_numpy(element).to(device=device, dtype=dtype)

    @classmethod
    def get_pytorch_bitwidth(cls, element: torch.Tensor) -> int:
        return cls.PYTORCH_DTYPE_TO_BITWIDTH[element.dtype]

    @classmethod
    def get_pytorch_dtype(cls, element: torch.Tensor) -> torch.dtype:
        return element.dtype

    @classmethod
    def get_pytorch_shape(cls, element: torch.Tensor) -> torch.Size:
        return element.size()
