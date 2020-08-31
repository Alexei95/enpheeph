import abc


class DeviceABC(abc.ABC):
    @property
    @abc.abstractmethod
    def weight_fault_probability(self):
        pass

    @property
    @abc.abstractmethod
    def activation_fault_probability(self):
        pass
