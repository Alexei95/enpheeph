import src.utils.mixins.dispatcher
import src.fi.injection.injectioncallback
import src.fi.utils.enums.parametertype
import src.fi.utils.enums.binaryfaultmaskop
import src.fi.utils.mixins.injectionhandlers.binaryhandler
import src.fi.utils.mixins.converters.numpyconverter
import src.fi.utils.mixins.converters.pytorchconverter


class WeightInjection(
        src.utils.mixins.dispatcher.Dispatcher,
        src.fi.utils.mixins.injectionhandlers.binaryhandler.BinaryHandler,
        src.fi.utils.mixins.converters.
                numpyconverter.NumpyConverter,
        src.fi.utils.mixins.converters.
                pytorchconverter.PyTorchConverter,
):
    @src.fi.injection.injectioncallback.InjectionCallback.register_decorator(
            src.fi.utils.enums.parametertype.ParameterType.Weight
    )
    @classmethod
    def init_weight_mask(
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

        # then we create the masks from the bit index
        binary_mask = cls.generate_fault_mask(
                bit_width=bitwidth,
                bit_index=fault.bit_index_conversion(
                            bit_index=fault.bit_index,
                            bit_width=bitwidth,
                ),
                endianness=fault.endianness,
                bit_value=endianness.bit_value,
        )


        numpy_weights = cls.single_pytorch_to_numpy(weights.flatten()[0])
        bit_index = fault.bit_index_conversion(fault.bit_index, fault.bit_width)
        mask = cls.generate_fault_mask()
        weights = src.fi.fiutils.inject_tensor_fault_pytorch(
                tensor=weights,
                fault=fault,
        )
        # we set the weights to the updated value
        setattr(module, fault.parameter_name, weights)
        # we return the new module
        return module

    @classmethod
    def pytorch_injection

    @classmethod
    def pytorch_and_mask_injection(cls, )


WeightInjection.register(
        name=src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp.AND,
        callable_=WeightInjection.pytorch_and_mask_injection
)
