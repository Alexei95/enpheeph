import abc


class InjectionModuleABC(abc.ABC):
    def __init__(self, operator, index, *args, **kwargs):
        self._operator = operator
        self._index = index
