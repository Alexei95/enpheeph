import functools
import typing

import torch

import src.fi.injection.injectioncallback
import src.fi.utils.enums.parametertype
import src.fi.utils.mixins.injectionhandlers.numpylikehandler
import src.fi.utils.mixins.converters.numpylikeconverter
import src.fi.utils.mixins.converters.pytorchdeviceawareconverter


# we map the class to the activation injection
@src.fi.injection.injectioncallback.InjectionCallback.register_decorator(
        src.fi.utils.enums.parametertype.ParameterType.DNNActivationDense
)
# NOTE: we can only have one of the following module per layer, as the parsing
# of the top module is done statically on the original structure, not on the
# updated layers
# NOTE: we cannot have dataclasses go with torch.nn.Module as Module.__init__
# must be called before the dataclasses init
# FIXME: implement also backward for fault-aware training
class DNNActivationDenseInjectionModule(
        torch.nn.Module,
        src.fi.utils.mixins.injectionhandlers.
                numpylikehandler.NumpyLikeHandler,
        src.fi.utils.mixins.converters.numpylikeconverter.NumpyLikeConverter,
        src.fi.utils.mixins.converters.
                pytorchdeviceawareconverter.PyTorchDeviceAwareConverter,
):
    def __init__(
                self,
                fault: 'src.fi.injection.faultdescriptor.FaultDescriptor',
                module: 'torch.nn.Module',
    ):
        super().__init__()

        self.fault = fault
        self.module = module
        self._mask = None

    def get_mask(
            self,
            output: torch.Tensor,
            library: str,
    ):
        # if the mask has been already defined, we return it
        if self._mask is not None:
            return self._mask

        # we get the required info
        tensor_shape = self.get_pytorch_shape(output)
        # we need to remove the batch size dimension, so that the mask is
        # usable on all possible batch sizes
        no_batch_size_tensor_shape = self.remove_pytorch_batch_size_from_shape(
                tensor_shape,
        )
        dtype = self.get_pytorch_dtype(output)
        bitwidth = self.get_pytorch_bitwidth(output)
        device = self.get_pytorch_device(output)

        # we convert them into numpy-like object
        numpy_like_dtype = self.pytorch_dtype_to_numpy_like_dtype(
                dtype=dtype,
                library=library,
        )
        numpy_like_device = self.pytorch_device_to_numpy_like_device(device)
        # we generate the mask for the injection
        numpy_like_mask = self.generate_fault_tensor_mask(
            dtype=numpy_like_dtype,
            bit_index=self.fault.bit_index_conversion(
                        bit_index=self.fault.bit_index,
                        bit_width=bitwidth,
            ),
            tensor_index=self.fault.tensor_index_conversion(
                    tensor_index=self.fault.tensor_index,
                    tensor_shape=no_batch_size_tensor_shape,
            ),
            tensor_shape=no_batch_size_tensor_shape,
            endianness=self.fault.endianness,
            bit_value=self.fault.bit_value,
            library=library,
            device=numpy_like_device,
        )

        self._mask = numpy_like_mask

        return self._mask

    def forward(self, x):
        # we get the exact result from the previous module
        y_temp = self.module(x)
        # we convert the output to numpy-like
        numpy_like_y_temp = self.pytorch_to_numpy_like(y_temp)
        # we use the converted element to get the library we are using
        numpy_like_string = self.get_numpy_like_string(numpy_like_y_temp)

        # we generate the mask
        mask = self.get_mask(
                output=y_temp,
                library=numpy_like_string,
        )

        # we inject the temporary output,
        numpy_like_y = self.inject_fault_tensor_from_mask(
                numpy_like_element=numpy_like_y_temp,
                mask=mask,
                in_place=True,
        )

        # we convert the injected final output from numpy-like to pytorch
        y = self.numpy_like_to_pytorch(
            numpy_like_y,
            dtype=self.get_pytorch_dtype(y_temp),
        )
        # we return the fault-injected tensor
        return y
