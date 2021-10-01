import torch

import enpheeph.injections.injectionabc


class OutputPyTorchFault(
        enpheeph.injections.injectionabc.InjectionABC,
        enpheeph.injections.mixins.pytorchmaskmixin.PyTorchMaskMixIn,
):
    def __init__(
            self,
            low_level_torch_plugin: enpheeph.injections.plugins.
            lowleveltorchmaskpluginabc,
    ):
        super().__init__()

        self.low_level_plugin = low_level_torch_plugin

        self.handle = None
        self.mask = None

    def output_fault_hook(self, module, input, output):
        self.generate_mask(output)

        masked_output = self.inject_mask()

        return masked_output

    def setup(self, module):
        self.handle = module.register_forward_hook(self.output_fault_hook)

        return module

    def teardown(self, module):
        self.handle.remove()

        self.handle = None
        self.mask = None

        return module