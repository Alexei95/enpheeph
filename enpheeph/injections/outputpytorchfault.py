import torch

import enpheeph.faults.pytorchfaultabc


class OutputPyTorchFault(
        enpheeph.faults.pytorchfaultabc.PyTorchFaultABC,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._mask = None

    def output_fault_hook(self, module, input, output):
        if self._mask is None:
            self.generate_mask(output)

        masked_output = 

        return masked_output

    def setup(self, module):
        
        

    def teardown(self, module):
        pass