from . import deviceabc


class GPUDevice(deviceabc.DeviceABVC):
    def __init__(self,
                 total_chip_area,
                 number_sub_components,
                 register_area_per_subcomponent,
                 memory_area_per_subcomponent, ):

    @property
    @abc.abstractmethod
    def weight_fault_probability(self):
        pass

    @property
    @abc.abstractmethod
    def activation_fault_probability(self):
        pass
