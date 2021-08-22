import abc


class CollateABC(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, batch, *args, **kwargs):
        input_, target = batch
        return self.call(input_, target)

    @abc.abstractmethod
    def call(self, input_, target):
        pass
