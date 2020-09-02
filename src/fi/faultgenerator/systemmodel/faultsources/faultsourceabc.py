import abc
import typing

from . import common


class FaultSourceABC(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def generate_fault_sources(self, *args, **kwargs) -> typing.Tuple:
        pass
