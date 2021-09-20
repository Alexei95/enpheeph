class FaultABC:
    def __init__(self, fault_locator):
        self.fault_locator = fault_locator

    def setup(self, module):
        pass

    def teardown(self, module):
        pass


class PyTorchFaultABC(FaultABC):
    def make_mask(self, tensor):
        pass


class ActivationPyTorchFault(PyTorchFaultABC):
    pass
