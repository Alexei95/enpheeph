import abc

import torch

import enpheeph.faults.faultabc
import enpheeph.injections.pytorchinjectionabc


class PyTorchFaultABC(
        enpheeph.faults.faultabc.FaultABC,
        enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC,
):
    @abc.abstractmethod
    def make_mask(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError