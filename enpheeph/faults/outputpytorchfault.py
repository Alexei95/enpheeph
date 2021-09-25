import torch

import enpheeph.injections.injectionabc


class OutputPyTorchFault(
        enpheeph.injections.injectionabc.InjectionABC,
):
    def __init__(self, fault_locator):
        self.fault_locator = fault_locator

    def setup(self, module):
        pass

    def teardown(self, module):
        pass