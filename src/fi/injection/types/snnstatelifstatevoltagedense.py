import functools
import inspect
import typing

import norse
import torch

import src.fi.injection.injectioncallback
import src.fi.utils.enums.parametertype
import src.fi.utils.mixins.injectionhandlers.numpylikehandler
import src.fi.utils.mixins.converters.numpylikeconverter
import src.fi.utils.mixins.converters.pytorchdeviceawareconverter


# we map the class to the activation injection
@src.fi.injection.injectioncallback.InjectionCallback.register_decorator(
        src.fi.utils.enums.parametertype.
        ParameterType.SNNStateLIFStateVoltageDense
)
# NOTE: we can only have one of the following module per layer, as the parsing
# of the top module is done statically on the original structure, not on the
# updated layers
# NOTE: we cannot have dataclasses go with torch.nn.Module as Module.__init__
# must be called before the dataclasses init
# FIXME: implement also backward for fault-aware training
class SNNStateLIFStateVoltageDenseInjectionModule(
        torch.nn.Module,
        src.fi.utils.mixins.injectionhandlers.
        numpylikehandler.NumpyLikeHandler,
        src.fi.utils.mixins.converters.numpylikeconverter.NumpyLikeConverter,
        src.fi.utils.mixins.converters.
        pytorchdeviceawareconverter.PyTorchDeviceAwareConverter,
):
    ARGS_KWARGS_NAMES = ('args', 'kwargs')
    EXTRA_SNN_PARAMETERS = ('dt', )
    EXTRA_RECURRENT_SNN_PARAMETERS = ('dt', 'autapses')

    def __init__(
                self,
                fault: 'src.fi.injection.faultdescriptor.FaultDescriptor',
                module: typing.Union['norse.torch.SNN', 'norse.torch.SNNCell'],
    ):
        super().__init__()

        if fault.sequence_time_step is None:
            raise ValueError(
                    "'sequence_time_step' value in FaultDescriptor "
                    "must be provided for SNN injection"
            )

        self.fault = fault
        self.old_module = module
        self._mask = None
        # this flag is used to check whether we need to run the for loop for
        # going through the simulation
        # we need to cover the simulation if the targeted module is
        # a non-Cell module, as in that case the simulation steps are handled
        # internally
        self.converted = False

        # this counter is used to remember the time step during forward
        self._counter = 0
        # we also save the old state at the end of forward so that
        # we reset the timer when the new state is different
        self._old_state = None

        # if the module is already a cell we save it
        if isinstance(module, norse.torch.module.snn.SNNCell):
            self.new_module = module
            self.converted = False
        # otherwise we need to convert it
        elif isinstance(module, norse.torch.module.snn.SNN):
            # we get all the strings for the arguments of the module init
            arguments = inspect.signature(module.__init__).parameters.keys()
            # we remove the extra keys and we add some arguments which
            # are not included
            filtered_arguments = [
                    x
                    for x in tuple(arguments) + self.EXTRA_SNN_PARAMETERS
                    if x not in self.ARGS_KWARGS_NAMES
            ]
            # we get the corresponding argument value from the module and
            # create a config dict
            config = {
                arg_name: getattr(module, arg_name)
                for arg_name in filtered_arguments
            }
            # we get the name of the class of the module
            module_class_name = module.__class__.__qualname__
            # we add Cell to convert the name
            new_module_class_name = module_class_name + 'Cell'
            # we get the corresponding new Cell module from norse
            new_module_class = getattr(norse.torch, new_module_class_name)
            # we create the new module
            new_module = new_module_class(**config)
            # we save the new module
            self.new_module = new_module
            self.converted = True
        # if it is not an SNN then we raise ValueError
        else:
            raise ValueError("This module is not supported")

    def get_mask(
            self,
            tensor: torch.Tensor,
            library: str,
    ):
        # if the mask has been already defined, we return it
        if self._mask is not None:
            return self._mask

        # we get the required info
        tensor_shape = self.get_pytorch_shape(tensor)
        # we need to remove the batch size dimension, so that the mask is
        # usable on all possible batch sizes
        no_batch_size_tensor_shape = self.remove_pytorch_batch_size_from_shape(
                tensor_shape,
        )
        dtype = self.get_pytorch_dtype(tensor)
        bitwidth = self.get_pytorch_bitwidth(tensor)
        device = self.get_pytorch_device(tensor)

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

    def forward(self, x, state=None):
        # if current state is None means we are starting from scratch
        # we also reset the timer if the saved state from the previous
        # iteration is different from the passed state, meaning non-continuity
        if state is None or self._old_state != state:
            self._counter = 0
            # if state is None we generate a fallback state from the current
            # input
            # this is done inside the target module, but we need the state
            # here for possible injection
            if state is None:
                state = self.new_module.state_fallback(x)

        # we consider the first index of the fault injection to be the time
        # step
        sequence_time_step_index = self.fault.bit_index_conversion(
            bit_index=self.fault.sequence_time_step_index,
            bit_width=self.counter,
        )
        if self.counter in sequence_time_step_index:
            # we select the target for injection
            if (src.fi.utils.enums.parametertype.
                    ParameterType.Voltage in self.fault.parameter_type
            ):
            target =
            mask = self.get_mask(
                    tensor=state,
                    library=numpy
            )




        # we get the exact result from the previous module
        y_temp = self.module(x)
        # we convert the output to numpy-like
        numpy_like_y_temp = self.pytorch_to_numpy_like(y_temp)
        # we use the converted element to get the library we are using
        numpy_like_string = self.get_numpy_like_string(numpy_like_y_temp)

        # we generate the mask
        mask = self.get_mask(
                tensor=y_temp,
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
