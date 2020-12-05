import abc

class SourceABC(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def generate_strikes(self, *args, **kwargs):
        pass
