try:
    import cupy
except ImportError:
    cupy = None

try:
    import numpy
except ImportError:
    numpy = None

import torch
import torch.utils.dlpack

CPU_TYPE = 'cpu'
CUDA_TYPE = 'cuda'


class PyTorchTensorWrapper(object):
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    @property
    def converted_tensor(self):
        if self._tensor.device.type == CPU_TYPE:
            if numpy is None:
                raise RuntimeError("numpy must be installed for this action")
            return self._tensor.numpy()
        elif self._tensor.device.type == CUDA_TYPE:
            if cupy is None:
                raise RuntimeError("cupy must be installed for this action")
            return cupy.fromDlpack(torch.utils.dlpack.to_dlpack(self._tensor))
