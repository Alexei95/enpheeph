import torch

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
                    enpheeph.injections.plugins.
                    lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
            ),
    ):
        super().__init__()

        self.fault_location = fault_location
        self.low_level_plugin = low_level_torch_plugin

        self.handle = None
        self.mask = None

    @property
    def module_name(self):
        return self.fault_location.injection_location.module_name

    def output_fault_hook(self, module, input, output):
        self.generate_mask(output)

        masked_output = self.inject_mask(output)

        return masked_output

    def setup(self, module):
        self.handle = module.register_forward_hook(self.output_fault_hook)

        return module

    def teardown(self, module):
        self.handle.remove()

        self.handle = None
        self.mask = None

        return module