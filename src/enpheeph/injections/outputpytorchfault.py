# -*- coding: utf-8 -*-
import typing

import enpheeph.injections.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmaskmixin
import enpheeph.injections.plugins.lowleveltorchmaskpluginabc
import enpheeph.utils.data_classes


class OutputPyTorchFault(
    enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC,
    enpheeph.injections.mixins.pytorchmaskmixin.PyTorchMaskMixIn,
):
    def __init__(
        self,
        fault_location: enpheeph.utils.data_classes.FaultLocation,
        low_level_torch_plugin: (
            enpheeph.injections.plugins.lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
        ),
    ):
        super().__init__()

        self.fault_location = fault_location
        self.low_level_plugin = low_level_torch_plugin

        self.handle = None
        self.mask = None

    @property
    def module_name(self) -> str:
        return self.fault_location.module_name

    def output_fault_hook(
        self,
        module: "torch.nn.Module",
        input: typing.Union[typing.Tuple["torch.Tensor"], "torch.Tensor"],
        output: "torch.Tensor",
    ) -> "torch.Tensor":
        self.generate_mask(output)

        masked_output = self.inject_mask(output)

        return masked_output

    def setup(self, module: "torch.nn.Module",) -> "torch.nn.Module":
        self.handle = module.register_forward_hook(self.output_fault_hook)

        return module

    def teardown(self, module: "torch.nn.Module",) -> "torch.nn.Module":
        self.handle.remove()

        self.handle = None
        self.mask = None

        return module