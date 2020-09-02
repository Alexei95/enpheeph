from . import deviceabc


class GPUDevice(deviceabc.DeviceABVC):
    def __init__(self,
                 flops,
                 execution_order,
                 *args, **kwargs):
        self._flops = flops
        self._execution_order = execution_order

    @property
    def execution_order(self):
        return self._execution_order

    def execution_time(self, n_operations):
        return n_operations / self._flops
