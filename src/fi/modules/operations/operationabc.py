import abc

import torch


class OperationABC(torch.nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
