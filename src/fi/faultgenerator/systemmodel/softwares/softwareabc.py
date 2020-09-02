import abc


class SoftwareABC(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def convert_faults_to_injection_modules(self, *args, **kwargs):
        pass
