import abc

import torch


class TransformABC(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if kwargs.get('events', None) is None:
            element = args[0]
            return self.call(element)
        else:
            element = kwargs['events']
            images = kwargs.get('images', None)
            return self.call(element), images

    @abc.abstractmethod
    def call(self, element):
        pass
