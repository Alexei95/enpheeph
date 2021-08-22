import copy
import typing

import torch

import src.utils.mixins.classutils
import src.utils.mixins.dispatcher
import src.fi.utils.mixins.converters.pytorchconverter


class PyTorchDeviceAwareConverter(
        src.utils.mixins.classutils.ClassUtils,
        src.utils.mixins.dispatcher.Dispatcher,
        src.fi.utils.mixins.converters.pytorchconverter.PyTorchConverter,
):
    # FIXME: check whether these methods can be moved
    # this method is used to remove all the associations after calling a
    # classmethod
    # in our case we cover the gpu and cpu devices
    @classmethod
    def _deregister_devices(cls):
        for device in cls.PYTORCH_SUPPORTED_DEVICES:
            cls.deregister(device)

    # this method is used to remove all the associations after calling a
    # classmethod, made especially for the libraries to Pytorch associations
    @classmethod
    def _deregister_libraries(cls):
        for string in cls.PYTORCH_SUPPORTED_STRINGS:
            cls.deregister(string)

    # this method converts to a numpy-like library a PyTorch tensor
    # it takes into account the device, and forces it to be in-place
    @classmethod
    def pytorch_to_numpy_like(
            cls,
            element: torch.Tensor,
    ) -> typing.Union['numpy.ndarray', 'cupy.ndarray']:
        cls.register_string_methods(
                cls,
                'pytorch_to_{}',
                string_list=cls.PYTORCH_SUPPORTED_STRINGS,
                name_list=cls.PYTORCH_SUPPORTED_DEVICES,
                error_if_none=True,
        )

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_pytorch_device_type(cls.get_pytorch_device(element)),
                element=element,
                in_place=True
        )

        cls._deregister_devices()

        return out

    # we convert from numpy to torch tensor, with optional device and type
    @classmethod
    def numpy_like_to_pytorch(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            dtype: torch.dtype,
    ) -> torch.Tensor:
        cls.register_string_methods(
                cls,
                '{}_to_pytorch',
                string_list=cls.PYTORCH_SUPPORTED_STRINGS,
                name_list=cls.PYTORCH_SUPPORTED_STRINGS,
                error_if_none=True,
        )

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
                dtype=dtype,
                in_place=True
        )

        cls._deregister_libraries()

        return out

    # this method converts the device, returning 'cpu' if it is not on CUDA
    @classmethod
    def pytorch_device_to_numpy_like_device(
            cls,
            device: torch.device
    ) -> typing.Union['cupy.cuda.Device', str]:
        if device == cls.PYTORCH_CPU_DEVICE:
            return 'cpu'
        elif cls.PYTORCH_CUPY_STRING is not None:
            return cls.pytorch_device_to_cupy_device(device)
        else:
            raise ValueError('cupy requested but not installed')

    # we can use this method for checking the availability of CUDA
    @classmethod
    def cuda_support(cls):
        return cls.PYTORCH_GPU_DEVICE is not None

    @classmethod
    def pytorch_dtype_to_numpy_like_dtype(
            cls,
            dtype: torch.dtype,
            library: str,
    ) -> typing.Union['numpy.dtype', 'cupy.dtype']:
        cls.register_string_methods(
                cls,
                'pytorch_dtype_to_{}_dtype',
                string_list=cls.PYTORCH_SUPPORTED_STRINGS,
                name_list=cls.PYTORCH_SUPPORTED_STRINGS,
                error_if_none=True,
        )

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                library,
                dtype=dtype,
        )

        cls._deregister_libraries()

        return out
