import copy

# fallback if cupy is not installed
try:
    import cupy
except ImportError:
    cupy = None
# PyTorch depends on Numpy, so we don't need this check, but we leave it
# to give an example on how to handle cross-library dependencies
try:
    import numpy
except ImportError:
    numpy = None
import torch
import torch.utils.dlpack


class PyTorchConverter(object):
    # NOTE: some types are equivalent, so the actual key is overwritten in the
    # dict, but they are left for reference
    PYTORCH_DTYPE_TO_BITWIDTH = {
            # 32-bit float
            torch.float32: 32,
            torch.float: 32,
            # 64-bit float
            torch.float64: 64,
            torch.double: 64,
            # 16-bit float
            torch.float16: 16,
            torch.half: 16,
            # brain float
            torch.bfloat16: 16,
            # 16-bit double floating point for complex
            torch.complex32: 32,
            # 32-bit double floating point for complex
            torch.complex64: 64,
            # 64-bit double floating point for complex
            torch.complex128: 128,
            # 64-bit double floating point for complex
            torch.cdouble: 128,
            # 8-bit unsigned integer
            torch.uint8: 8,
            # 8-bit signed integer
            torch.int8: 8,
            # 16-bit signed integer
            torch.int16: 16,
            torch.short: 16,
            # 32-bit signed integer
            torch.int32: 32,
            torch.int: 32,
            # 64-bit signed integer
            torch.int64: 64,
            torch.long: 64,
            # boolean
            torch.bool: 1,
    }
    # NOTE: here we assume there is no fallback option, so the correct one
    # must be utilized from the beginning
    if numpy is not None:
        # generated using
        # for key in pytorch_dtype_list.keys():
        #     try:
        #         dict_[key] = torch.tensor(1, dtype=key).numpy().dtype
        #     except Exception as e:
        #         print('Error:', e)
        PYTORCH_DTYPE_TO_NUMPY_DTYPE = {
                # 32-bit float
                torch.float32: numpy.dtype('float32'),
                torch.float: numpy.dtype('float32'),
                # 64-bit float
                torch.float64: numpy.dtype('float64'),
                torch.double: numpy.dtype('float64'),
                # 16-bit float
                torch.float16: numpy.dtype('float16'),
                torch.half: numpy.dtype('float16'),
                # brain float, bfloat16, are not supported by numpy
                # 16-bit double floating point for complex, complex32, are not
                # supported by numpy
                # 32-bit double floating point for complex
                torch.complex64: numpy.dtype('complex64'),
                # 64-bit double floating point for complex
                torch.complex128: numpy.dtype('complex128'),
                torch.cdouble: numpy.dtype('complex128'),
                # 8-bit unsigned integer
                torch.uint8: numpy.dtype('uint8'),
                # 8-bit signed integer
                torch.int8: numpy.dtype('int8'),
                # 16-bit signed integer
                torch.int16: numpy.dtype('int16'),
                torch.short: numpy.dtype('int16'),
                # 32-bit signed integer
                torch.int32: numpy.dtype('int32'),
                torch.int: numpy.dtype('int32'),
                # 64-bit signed integer
                torch.int64: numpy.dtype('int64'),
                # boolean
                torch.bool: numpy.dtype('bool'),
        }
        PYTORCH_NUMPY_STRING = 'numpy'
        PYTORCH_CPU_DEVICE = torch.device('cpu')
        PYTORCH_NUMPY_DEVICE = PYTORCH_CPU_DEVICE.type
    else:
        PYTORCH_DTYPE_TO_NUMPY_DTYPE = {}
        PYTORCH_NUMPY_STRING = None
        PYTORCH_CPU_DEVICE = None
        PYTORCH_NUMPY_DEVICE = None
    # if cupy or cuda are not available then we skip this part
    if cupy is not None and torch.cuda.is_available():
        # generated using
        # for key in pytorch_dtype_list.keys():
        #     try:
        #         dict_[key] = cupy.dtype(
        #                 torch.tensor(1, dtype=key).numpy().dtype)
        #         )
        #     except Exception as e:
        #         print('Error:', e)
        PYTORCH_DTYPE_TO_CUPY_DTYPE = {
                # 32-bit float
                torch.float32: cupy.dtype('float32'),
                torch.float: cupy.dtype('float32'),
                # 64-bit float
                torch.float64: cupy.dtype('float64'),
                torch.double: cupy.dtype('float64'),
                # 16-bit float
                torch.float16: cupy.dtype('float16'),
                torch.half: cupy.dtype('float16'),
                # brain float, bfloat16, are not supported by cupy
                # 16-bit double floating point for complex, complex32, are not
                # supported by cupy
                # 32-bit double floating point for complex
                torch.complex64: cupy.dtype('complex64'),
                # 64-bit double floating point for complex
                torch.complex128: cupy.dtype('complex128'),
                torch.cdouble: cupy.dtype('complex128'),
                # 8-bit unsigned integer
                torch.uint8: cupy.dtype('uint8'),
                # 8-bit signed integer
                torch.int8: cupy.dtype('int8'),
                # 16-bit signed integer
                torch.int16: cupy.dtype('int16'),
                torch.short: cupy.dtype('int16'),
                # 32-bit signed integer
                torch.int32: cupy.dtype('int32'),
                torch.int: cupy.dtype('int32'),
                # 64-bit signed integer
                torch.int64: cupy.dtype('int64'),
                # boolean
                torch.bool: cupy.dtype('bool'),
        }
        PYTORCH_CUPY_STRING = 'cupy'
        PYTORCH_GPU_DEVICE = torch.device('cuda')
        PYTORCH_CUPY_DEVICE = PYTORCH_GPU_DEVICE.type
    else:
        PYTORCH_DTYPE_TO_CUPY_DTYPE = {}
        PYTORCH_CUPY_STRING = None
        PYTORCH_GPU_DEVICE = None
        PYTORCH_CUPY_DEVICE = None

    PYTORCH_STRINGS = (PYTORCH_CUPY_STRING, PYTORCH_NUMPY_STRING)
    PYTORCH_SUPPORTED_STRINGS = tuple(
            filter(lambda x: x is not None, PYTORCH_STRINGS)
    )
    PYTORCH_DEVICES = (PYTORCH_CUPY_DEVICE, PYTORCH_NUMPY_DEVICE)
    PYTORCH_SUPPORTED_DEVICES = tuple(
            filter(lambda x: x is not None, PYTORCH_DEVICES)
    )

    @classmethod
    def pytorch_to_cupy(
            cls,
            element: torch.Tensor,
            # if in_place is True, the default, we do not copy the tensor
            in_place: bool = True,
    ) -> 'cupy.ndarray':
        if cls.PYTORCH_CUPY_STRING is None:
            raise NotImplementedError(
                    'cupy is not available, '
                    'function not supported'
            )
        if in_place:
            return cupy.fromDlpack(
                    torch.utils.dlpack.to_dlpack(element.squeeze())
            )
        else:
            return cupy.fromDlpack(
                    torch.utils.dlpack.to_dlpack(element.copy().squeeze())
            )

    @classmethod
    def pytorch_to_numpy(
            cls,
            element: torch.Tensor,
            # if in_place is True, the default, we do not copy the tensor
            in_place: bool = True,
    ) -> 'numpy.ndarray':
        if cls.PYTORCH_NUMPY_STRING is None:
            raise NotImplementedError(
                    'numpy is not available, '
                    'function not supported'
            )
        if in_place:
            return element.squeeze().cpu().numpy()
        else:
            return element.copy().squeeze().cpu().numpy()

    # we convert from numpy to torch tensor, with optional device and type
    @classmethod
    def numpy_to_pytorch(
            cls,
            element: 'numpy.ndarray',
            *,
            # dtype and device are neglected if in_place is true
            dtype: torch.dtype = None,
            device: torch.device = None,
            # this flag is used to indicate whether to copy the array in memory
            # or to act on the same memory location, creating only a new Python
            # object
            in_place: bool = True,
    ) -> torch.Tensor:
        if cls.PYTORCH_NUMPY_STRING is None:
            raise NotImplementedError(
                    'numpy is not available, '
                    'function not supported'
            )
        if in_place:
            return torch.from_numpy(element)
        else:
            return torch.from_numpy(element).to(device=device, dtype=dtype)

    # we convert from cupy to torch tensor, with optional device and type
    @classmethod
    def cupy_to_pytorch(
            cls,
            element: 'cupy.ndarray',
            *,
            # dtype and device are neglected if in_place is true
            dtype: torch.dtype = None,
            device: torch.device = None,
            # this flag is used to indicate whether to copy the array in memory
            # or to act on the same memory location, creating only a new Python
            # object
            in_place: bool = True,
    ) -> torch.Tensor:
        if cls.PYTORCH_CUPY_STRING is None:
            raise NotImplementedError(
                    'cupy is not available, '
                    'function not supported'
            )
        if in_place:
            tensor = torch.utils.dlpack.from_dlpack(element.toDlpack())
        else:
            tensor = torch.utils.dlpack.from_dlpack(
                    element.toDlpack()).to(device=device, dtype=dtype)
        return tensor

    @classmethod
    def get_pytorch_bitwidth(cls, element: torch.Tensor) -> int:
        return cls.PYTORCH_DTYPE_TO_BITWIDTH[element.dtype]

    @classmethod
    def get_pytorch_dtype(cls, element: torch.Tensor) -> torch.dtype:
        return element.dtype

    @classmethod
    def get_pytorch_shape(cls, element: torch.Tensor) -> torch.Size:
        return element.size()

    @classmethod
    def remove_pytorch_batch_size_from_shape(
            cls,
            size: torch.Size
    ) -> torch.Size:
        # FIXME: this method should be able to remove a custom dimension
        # or use other way to recognize the correct one to remove, e.g.
        # named tensors or similar other tricks
        # for now we only remove the first dimension
        return size[1:]

    @classmethod
    def pytorch_dtype_to_numpy_dtype(cls, dtype: torch.dtype) -> 'numpy.dtype':
        if numpy is None:
            raise NotImplementedError(
                    'numpy is not available, '
                    'function not supported'
            )
        return cls.PYTORCH_DTYPE_TO_NUMPY_DTYPE[dtype]

    @classmethod
    def pytorch_dtype_to_cupy_dtype(cls, dtype: torch.dtype) -> 'cupy.dtype':
        if cupy is None:
            raise NotImplementedError(
                    'cupy is not available, '
                    'function not supported'
            )
        return cls.PYTORCH_DTYPE_TO_CUPY_DTYPE[dtype]

    @classmethod
    def get_pytorch_device(cls, element: torch.Tensor) -> torch.device:
        return element.device

    @classmethod
    def get_pytorch_device_type(cls, device: torch.device) -> str:
        return device.type

    @classmethod
    def pytorch_device_to_cupy_device(
            cls,
            device: torch.device
    ) -> 'cupy.cuda.Device':
        if cls.PYTORCH_CUPY_STRING is None:
            raise NotImplementedError(
                    'cupy is not available, '
                    'function not supported'
            )

        return cupy.cuda.Device(device.index)

    @classmethod
    def cupy_device_to_pytorch_device(
            cls,
            device: 'cupy.cuda.Device'
    ) -> torch.device:
        if cls.PYTORCH_CUPY_STRING is None:
            raise NotImplementedError(
                    'cupy is not available, '
                    'function not supported'
            )

        return torch.device('cuda', device.id)
