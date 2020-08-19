import abc

import torch


class OperationABC(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
