import functools
import inspect
import typing

import norse
import torch

import src.fi.injection.injectioncallback
import src.fi.utils.enums.parametertype
import src.fi.utils.mixins.injectionhandlers.numpylikehandler
import src.fi.utils.mixins.converters.numpylikeconverter
import src.fi.utils.mixins.converters.norsedeviceawareconverter


# we map the class to the SNN LIF state voltage injection
# only for dense arrays
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
        norsedeviceawareconverter.NorseDeviceAwareConverter,
):
    ARGS_KWARGS_NAMES = ('args', 'kwargs')
    EXTRA_SNN_PARAMETERS = ('dt', )
    EXTRA_RECURRENT_SNN_PARAMETERS = ('dt', 'autapses')
    CELL_STRING = 'Cell'

    def __init__(
                self,
                fault: 'src.fi.injection.faultdescriptor.FaultDescriptor',
                module: typing.Union['norse.torch.SNN', 'norse.torch.SNNCell'],
    ):
        super().__init__()

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
            new_module_class_name = module_class_name + self.CELL_STRING
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
            tensor: torch.Tensor
    ):
        # if the mask has been already defined, we return it
        if self._mask is not None:
            return self._mask

        # we get the library from the tensor
        library = self.get_numpy_like_string(self.pytorch_to_numpy_like(
                tensor
        ))

        # we get the required info
        tensor_shape = self.get_pytorch_shape(tensor)
        # here we need to remove only the batch size, as the time-step
        # dimension is already not present, since its state represents
        # the neuronal state at a single time-step
        pure_tensor_shape = self.remove_pytorch_batch_size_from_shape(
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
            # here we use a fake first index as it covers the sequence time
            # step, which is not useful in this case
            tensor_index=self.remove_norse_sequence_time_step_from_shape(
                    # we need to select the correct indices, as the first
                    # one is the time step that we do not need when making
                    # the mask
                    # only the actual tensor shape is required, so we select
                    # the shape from the index excluding the first element
                    # related to the time step
                    self.fault.tensor_index_conversion(
                            tensor_index=self.fault.tensor_index,
                            tensor_shape=[0, *pure_tensor_shape],
                    )
            ),
            tensor_shape=pure_tensor_shape,
            endianness=self.fault.endianness,
            bit_value=self.fault.bit_value,
            library=library,
            device=numpy_like_device,
        )

        self._mask = numpy_like_mask

        return self._mask

    def _forward_single_step(self, x, state=None):
        # if current state is None means we are starting from scratch
        if state is None:
            self._counter = 0
            # we generate a fallback state from the current input
            # this is done inside the target module, but we need the state
            # here for possible injection
            state = self.new_module.state_fallback(x)

        # we consider the first index of the fault injection to be the time
        # step
        # to convert the index we pass the fault index together with the
        # current length of the state sequence
        # we can use the current counter as for each iteration we are
        # considering only the current time step
        # we are interested only in the time step index
        # we force the function to return a list of indices, and not slices
        # for the shape, we need to increase the counter by 1, as if the
        # counter is 0 the corresponding shape will be 1 and so on
        # counter works as an index in a list, and we need its length here
        time_tensor_index = (
                self.fault.tensor_index
                if isinstance(self.fault.tensor_index, type(Ellipsis))
                else [
                        self.remove_norse_sequence_time_step_from_shape(
                                self.fault.tensor_index
                        )
                ]
        )
        # we need to get the first index using the interface
        # the returned type is a tuple with the indices
        sequence_time_step_index = self.\
        get_norse_sequence_time_step_from_index(
                self.fault.tensor_index_conversion(
                    tensor_index=time_tensor_index,
                    tensor_shape=[self._counter + 1],
                    force_index=True,
                )
        )
        if self._counter in sequence_time_step_index:
            state_variable = None
            state_variable_name = None

            # we select the target for injection
            if (
                    src.fi.utils.enums.parametertype.
                    ParameterType.Voltage in self.fault.parameter_type
            ):
                state_variable = state.v
                state_variable_name = 'v'

            # if the target is not None we execute the injection
            if state_variable is not None and state_variable_name is not None:
                # we convert the state_variable from the state to numpy-like
                numpy_like_state_variable = self.pytorch_to_numpy_like(
                        state_variable
                )

                mask = self.get_mask(
                        tensor=state_variable,
                )

                # we inject the temporary output,
                numpy_like_state_variable_injected = self.\
                        inject_fault_tensor_from_mask(
                                numpy_like_element=numpy_like_state_variable,
                                mask=mask,
                                in_place=True,
                        )

                # we convert the injected final output from numpy-like to
                # pytorch
                state_variable_injected = self.numpy_like_to_pytorch(
                    numpy_like_state_variable_injected,
                    dtype=self.get_pytorch_dtype(state_variable),
                )

                # we update the state variable value in the state
                # it cannot be done directly as the state is a tuple, so we
                # need to create a new one substituting the state variable
                state = state._replace(**{
                        state_variable_name: state_variable_injected
                })

        # we increase the counter
        self._counter += 1
        # here we simply run the forward
        return self.new_module(x, state)

    def forward(self, x, state=None):
        if self.converted:
            time_sequence_length = x[0]
            y = [None] * time_sequence_length
            for ts in range(time_sequence_length):
                y[ts], state = self._forward_single_step(x[ts], state)
            return torch.stack(y, dim=0), state
        else:
            return self._forward_single_step(x, state)
