import typing

import torch

import src.utils.mixins.dispatcher
import src.fi.injection.injectioncallback
import src.fi.utils.enums.parametertype
import src.fi.utils.enums.binaryfaultmaskop
import src.fi.utils.mixins.injectionhandlers.binaryhandler
import src.fi.utils.mixins.converters.cupyconverter
import src.fi.utils.mixins.converters.numpyconverter
import src.fi.utils.mixins.converters.pytorchdeviceawareconverter


class WeightInjection(
        # not required if we use PyTorchDeviceAwareConverter, otherwise we
        # get MRO error
        # src.utils.mixins.dispatcher.Dispatcher,
        src.fi.utils.mixins.injectionhandlers.binaryhandler.BinaryHandler,
        src.fi.utils.mixins.converters.cupyconverter.CupyConverter,
        src.fi.utils.mixins.converters.numpyconverter.NumpyConverter,
        src.fi.utils.mixins.converters.
                pytorchdeviceawareconverter.PyTorchDeviceAwareConverter,
):
    @classmethod
    def init_weight_mask_pytorch(
            cls,
            fault: 'src.fi.injection.faultdescriptor.FaultDescriptor',
            module: 'torch.nn.Module') -> 'torch.nn.Module':
        # we get the weights
        weights = getattr(module, fault.parameter_name)

        # we generate the mask for the weights
        # here we only need the tensor shape, the dtype and the bitwidth
        tensor_shape = cls.get_pytorch_shape(weights)
        dtype = cls.get_pytorch_dtype(weights)
        bitwidth = cls.get_pytorch_bitwidth(weights)
        device = cls.get_pytorch_device(weights)
        device_type = cls.get_pytorch_device_type(device)

        # then we create the masks from the bit index
        binary_mask = cls.generate_fault_mask(
                bit_width=bitwidth,
                bit_index=fault.bit_index_conversion(
                        bit_index=fault.bit_index,
                        bit_width=bitwidth,
                ),
                endianness=fault.endianness,
                bit_value=fault.bit_value,
        )

        # we convert it into a numpy-like object mask
        # using the correct fill_value
        # to automatically do this depending on the device, we use the
        # classmethods inherited from PyTorchDeviceAwareConverter
        # first we use the call to get the converted dtype for the intermediate
        # library
        cls.register(cls.CUPY_DEVICE, cls.pytorch_dtype_to_cupy_dtype)
        cls.register(cls.NUMPY_DEVICE, cls.pytorch_dtype_to_numpy_dtype)
        # we dispatch the call
        numpy_like_dtype = cls.dispatch_call(
                device_type,
                dtype=dtype,
        )
        # we clear the associations
        cls._deregister_devices()

        # we need a dispatcher for the device conversion
        # here we do not use Numpy, so it will be a lambda returning the input
        # as the output will not be used
        cls.register(cls.CUPY_DEVICE, cls.pytorch_device_to_cupy_device)
        cls.register(cls.NUMPY_DEVICE, lambda x: x)
        # the device is required to create the array in the correct position
        # so that it can be converted to PyTorch on the correct GPU, without
        # memory transfers
        numpy_like_device = cls.dispatch_call(device_type, device)
        # we clear the associations
        cls._deregister_devices()

        # we create a new dispatcher for the binary mask broadcast
        cls.register(cls.CUPY_DEVICE, cls.binary_to_cupy_broadcast)
        cls.register(cls.NUMPY_DEVICE, cls.binary_to_numpy_broadcast)
        # we call the dispatcher with the current device
        numpy_like_mask = cls.dispatch_call(
                device_type,
                binary=binary_mask.mask,
                dtype=numpy_like_dtype,
                index=fault.tensor_index_conversion(
                        tensor_index=fault.tensor_index,
                        tensor_shape=tensor_shape,
                ),
                shape=tensor_shape,
                fill_value=binary_mask.fill_value,
                device=numpy_like_device,
        )
        # we clear the associations
        cls._deregister_devices()

        # we convert the weights to numpy-like
        numpy_like_weights = cls.pytorch_to_numpy_like(weights)

        # we need the original dtypes for both the numpy-like weights and mask
        # register classes
        cls.register(cls.CUPY_DEVICE, cls.get_cupy_dtype)
        cls.register(cls.NUMPY_DEVICE, cls.get_numpy_dtype)
        # we get the dtypes
        numpy_like_weights_dtype = cls.dispatch_call(
                device_type,
                element=numpy_like_weights,
        )
        numpy_like_mask_dtype = cls.dispatch_call(
                device_type,
                element=numpy_like_mask,
        )
        # we clear the associations
        cls._deregister_devices()

        # we register the classes for converting the numpy-like weights
        cls.register(cls.CUPY_DEVICE, cls.cupy_dtype_to_bitwise_cupy)
        cls.register(cls.NUMPY_DEVICE, cls.numpy_dtype_to_bitwise_numpy)
        # we convert the dtypes to uint... to have bitwise operations
        numpy_like_weights_bitwise = cls.dispatch_call(
                device_type,
                element=numpy_like_weights,
        )
        numpy_like_mask_bitwise = cls.dispatch_call(
                device_type,
                element=numpy_like_mask,
        )
        # we clear the associations
        cls._deregister_devices()

        # we use the mask in the other backend, as otherwise we would get
        # reinterpreted with PyTorch
        # we register the associations for the different mask operations
        cls.register(
                binary_mask.operation.AND,
                cls.numpy_like_and_mask_injection,
        )
        cls.register(
                binary_mask.operation.OR,
                cls.numpy_like_or_mask_injection,
        )
        cls.register(
                binary_mask.operation.XOR,
                cls.numpy_like_xor_mask_injection,
        )
        # we dispatch the call for injecting
        # the operation is done in-place, but the out array is also returned
        injected_numpy_like_weights_bitwise = cls.dispatch_call(
                binary_mask.operation,
                element=numpy_like_weights_bitwise,
                mask=numpy_like_mask_bitwise,
                in_place=True,
        )
        # we clear the associations
        cls._deregister_mask_types()

        # now we convert back to the original dtypes and finally to PyTorch
        # we register the classes for converting the numpy-like weights
        cls.register(cls.CUPY_DEVICE, cls.bitwise_cupy_to_cupy_dtype)
        cls.register(cls.NUMPY_DEVICE, cls.bitwise_numpy_to_numpy_dtype)
        # we convert back
        # also for the injected ones
        # even though most probably it is in the same memory area as the
        # original weights
        injected_numpy_like_weights = cls.dispatch_call(
                device_type,
                element=injected_numpy_like_weights_bitwise,
                dtype=numpy_like_weights_dtype,
        )
        numpy_like_weights = cls.dispatch_call(
                device_type,
                element=numpy_like_weights_bitwise,
                dtype=numpy_like_weights_dtype,
        )
        numpy_like_mask = cls.dispatch_call(
                device_type,
                element=numpy_like_mask_bitwise,
                dtype=numpy_like_mask_dtype
        )
        # we clear the associations
        cls._deregister_devices()

        # conversion from numpy-like to PyTorch
        injected_weights = cls.numpy_like_to_pytorch(
            element=injected_numpy_like_weights,
            dtype=dtype,
        )

        # we set the weights to the updated value
        setattr(module, fault.parameter_name, torch.nn.Parameter(injected_weights))
        # we return the new module
        return module

    @classmethod
    def _deregister_mask_types(cls):
        cls.deregister(
                src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp.AND
        )
        cls.deregister(
                src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp.OR
        )
        cls.deregister(
                src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp.XOR
        )

    @classmethod
    def numpy_like_and_mask_injection(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            mask: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            in_place: bool = True,
    ):
        # if the operation is in-place, we set the output of the operation
        # to be the input, otherwise it is None and it will create a new array
        if in_place:
            element &= mask
            out = element
        else:
            out = element & mask

        return out

    @classmethod
    def numpy_like_or_mask_injection(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            mask: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            in_place: bool = True,
    ):
        # if the operation is in-place, we set the output of the operation
        # to be the input, otherwise it is None and it will create a new array
        if in_place:
            element |= mask
            out = element
        else:
            out = element | mask

        return out

    @classmethod
    def numpy_like_xor_mask_injection(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            mask: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            in_place: bool = True,
    ):
        # if the operation is in-place, we set the output of the operation
        # to be the input, otherwise it is None and it will create a new array
        if in_place:
            element ^= mask
            out = element
        else:
            out = element ^ mask

        return out


src.fi.injection.injectioncallback.InjectionCallback.register(
        src.fi.utils.enums.parametertype.ParameterType.Weight,
        WeightInjection.init_weight_mask_pytorch
)
