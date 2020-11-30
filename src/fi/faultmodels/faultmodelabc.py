import abc


class FaultModelABC(object):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        return NotImplemented
