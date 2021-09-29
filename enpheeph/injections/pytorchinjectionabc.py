import abc

import typing


class PyTorchInjectionABC(abc.ABC):
    @abc.abstractmethod
    def setup(self, *args, **kwargs) -> typing.Any:
        return NotImplemented

    @abc.abstractmethod
    def teardown(self, *args, **kwargs) -> typing.Any:
        return NotImplemented

