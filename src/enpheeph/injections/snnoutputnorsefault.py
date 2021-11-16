# -*- coding: utf-8 -*-
# type: ignore[misc,assignment,name-defined,unreachable,union-attr,attr-defined,operator]
# flake8: noqa
# we ignore mypy/flake8 errors here as this injection needs to be refactored
import typing

import norse

import enpheeph.injections.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmaskmixin
import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.utils.data_classes


class SNNOutputNorseFault(
    enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC,
    enpheeph.injections.mixins.pytorchmaskmixin.PyTorchMaskMixin,
):
    def __init__(
        self,
        fault_location: enpheeph.utils.data_classes.FaultLocation,
        low_level_torch_plugin: (
            # black has issues with very long names
            # fmt: off
            enpheeph.injections.plugins.mask.
            lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
            # fmt: on
        ),
    ):
        super().__init__()

        if fault_location.time_index is None:
            raise ValueError("time_index must be passed in the injection for SNNs")

        self.fault_location = fault_location
        self.low_level_plugin = low_level_torch_plugin

        self.handle = None
        self.mask = None
        self.timestep_counter = None

    @property
    def module_name(self) -> str:
        return self.fault_location.module_name

    # this hook assumes that for each forward call, the initial state at the
    # first execution point is None
    # in this way we can count and locate precisely the timesteps, using only
    # the forward hook and without modifying the norse code
    # NOTE: it would not work if the initial state used as input is different
    # from None, so be careful
    def snn_output_fault_hook(
        self,
        module: "torch.nn.Module",
        input: typing.Union[typing.Tuple["torch.Tensor"], "torch.Tensor"],
        output: "torch.Tensor",
    ) -> "torch.Tensor":
        if input[1] is None:
            self.timestep_counter = 0
        elif isinstance(input[1], tuple):
            self.timestep_counter += 1
        else:
            raise RuntimeError("Not compatible with this way of calling")

        # find a way to check if we are in the index range
        # we simply check the different possibilities
        time_index = self.fault_location.time_index
        if isinstance(time_index, slice):
            index = range(time_index.start, time_index.stop, time_index.step)
        elif isinstance(time_index, typing.Sequence):
            index = time_index
        elif isinstance(time_index, type(Ellipsis)):
            index = range(self.timestep_counter + 1)
        elif isinstance(time_index, int):
            index = (time_index,)
        else:
            raise IndexError("Unsupported time_index for SNN fault injection")

        # if the current counter is in the index, then we inject the fault
        if self.timestep_counter in index:
            self.generate_mask(output)

            masked_output = self.inject_mask(output)

            return masked_output
        else:
            return output

    def setup(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        if not isinstance(module, norse.torch.module.snn.SNNCell):
            raise RuntimeError(
                "Currently SNN injection supports only SNNCell from norse"
            )
        self.handle = module.register_forward_hook(self.output_fault_hook)

        return module

    def teardown(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        self.handle.remove()

        self.handle = None
        self.mask = None

        return module
