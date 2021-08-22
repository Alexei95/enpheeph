import typing

import torch

import enpheeph.utils.mixins.dispatcher
import enpheeph.fi.injection.injectioncallback
import enpheeph.fi.utils.enums.parametertype
import enpheeph.fi.utils.enums.faultmaskop
import enpheeph.fi.utils.mixins.injectionhandlers.numpylikehandler
import enpheeph.fi.utils.mixins.converters.numpylikeconverter
import enpheeph.fi.utils.mixins.converters.pytorchdeviceawareconverter


class DNNWeightDenseInjection(
        # not required if we use PyTorchDeviceAwareConverter, otherwise we
        # get MRO error
        # enpheeph.utils.mixins.dispatcher.Dispatcher,
        enpheeph.fi.utils.mixins.injectionhandlers.
                numpylikehandler.NumpyLikeHandler,
        enpheeph.fi.utils.mixins.converters.numpylikeconverter.NumpyLikeConverter,
        enpheeph.fi.utils.mixins.converters.
                pytorchdeviceawareconverter.PyTorchDeviceAwareConverter,
):
    @classmethod
    def init_weight_mask_pytorch(
            cls,
            fault: 'enpheeph.fi.injection.faultdescriptor.FaultDescriptor',
            module: 'torch.nn.Module',
            in_place: bool = True,
    ) -> 'torch.nn.Module':
        # we get the weights
        weights = getattr(module, fault.parameter_name)

        # we generate the mask for the weights
        # here we only need the tensor shape, the dtype and the bitwidth
        tensor_shape = cls.get_pytorch_shape(weights)
        dtype = cls.get_pytorch_dtype(weights)
        bitwidth = cls.get_pytorch_bitwidth(weights)

        # we convert the weights to numpy-like
        numpy_like_weights = cls.pytorch_to_numpy_like(weights)

        # we inject the numpy-like object
        injected_numpy_like_weights = cls.inject_fault_tensor(
                numpy_like_element=numpy_like_weights,
                # we need to convert the bit indices
                bit_index=fault.bit_index_conversion(
                        bit_index=fault.bit_index,
                        bit_width=bitwidth,
                ),
                # we need to convert the tensor indices as well
                tensor_index=fault.tensor_index_conversion(
                        tensor_index=fault.tensor_index,
                        tensor_shape=tensor_shape,
                ),
                tensor_shape=tensor_shape,
                endianness=fault.endianness,
                bit_value=fault.bit_value,
                in_place=in_place,
        )

        # conversion from numpy-like to PyTorch not needed if in-place
        # if in-place, they share the same memory location, so all changes
        # are automatically propagated
        if not in_place:
            injected_weights = cls.numpy_like_to_pytorch(
                element=injected_numpy_like_weights,
                dtype=dtype,
            )
            setattr(
                    module,
                    fault.parameter_name,
                    torch.nn.Parameter(injected_weights)
            )
        # we return the new module
        return module


enpheeph.fi.injection.injectioncallback.InjectionCallback.register(
        enpheeph.fi.utils.enums.parametertype.ParameterType.DNNWeightDense,
        DNNWeightDenseInjection.init_weight_mask_pytorch
)
