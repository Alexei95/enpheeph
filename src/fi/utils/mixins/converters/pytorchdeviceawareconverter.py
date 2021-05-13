import copy
import typing

import torch

import src.utils.mixins.dispatcher
import src.fi.utils.mixins.converters.pytorchconverter


class PyTorchDeviceAwareConverter(
        src.utils.mixins.dispatcher.Dispatcher,
        src.fi.utils.mixins.converters.pytorchconverter.PyTorchConverter,
):
    # if cuda is not available we set the devices to a null value
    if torch.cuda.is_avalaible():
        GPU_DEVICE = torch.device('cuda')
        CUPY_DEVICE = GPU_DEVICE.type
    else:
        GPU_DEVICE = None
        CUPY_DEVICE = None

    CPU_DEVICE = torch.device('cpu')

    NUMPY_DEVICE = CPU_DEVICE.type

    NUMPY_STRING = 'numpy'
    CUPY_STRING = 'cupy'

    # this method is used to remove all the associations after calling a
    # classmethod
    # in our case we cover the gpu and cpu devices
    @classmethod
    def _deregister_devices(cls):
        cls.deregister(cls.CUPY_DEVICE)
        cls.deregister(cls.NUMPY_DEVICE)

    # this method is used to remove all the associations after calling a
    # classmethod, made especially for the libraries to Pytorch associations
    @classmethod
    def _deregister_libraries(cls):
        cls.deregister(cls.NUMPY_STRING)
        cls.deregister(cls.CUPY_STRING)

    # FIXME: check where to put this
    # this method returns the main library (the first module) associated to an
    # object, by going through its class, the module and then returning the
    # first module name
    @classmethod
    def get_main_library_from_object(cls, object: typing.Any):
        return object.__class__.__module__.split('.')[0]

    # this method converts to a numpy-like library a PyTorch tensor
    # it takes into account the device, and forces it to be in-place
    @classmethod
    def pytorch_to_numpy_like(
            cls,
            element: torch.Tensor,
    ) -> typing.Union['numpy.ndarray', 'cupy.ndarray']:
        cls.register(cls.CUPY_DEVICE, cls.pytorch_to_cupy)
        cls.register(cls.NUMPY_DEVICE, cls.pytorch_to_numpy)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_pytorch_device_type(element.device),
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
        cls.register(cls.CUPY_STRING, cls.cupy_to_pytorch)
        cls.register(cls.NUMPY_STRING, cls.numpy_to_pytorch)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
                dtype=dtype,
                in_place=True
        )

        cls._deregister_libraries()

        return out
