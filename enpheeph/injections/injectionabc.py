import abc


class InjectionABC(abc.ABC):
    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        return NotImplemented

    @abc.abstractmethod
    def teardown(self, *args, **kwargs):
        return NotImplemented

    @property
    @abc.abstractmethod
    def module_name(self):
        return NotImplemented


# only a stub as middle ground
class PyTorchInjectionABC(InjectionABC):
    pass


class ActivationPyTorchInjectionABC(PyTorchInjectionABC):
    def make_sequential_and_append(self, original_module, to_append_module):
        pass
