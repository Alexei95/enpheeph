import pathlib
import sys

import torch

# PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
# if str(PROJECT_DIR) not in sys.path:
#     sys.path.append(str(PROJECT_DIR))

from . import basefi

# using this fault injector for changing all the
class RandomTensorFI(torch.nn.Module, basefi.BaseFI):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # call init from BaseFI
        self.init(*args, **kwargs)

    def forward(self, x):
        if self.fi_enabled:
            r = torch.rand_like(x)
        else:
            r = x
        return r

FAULT_INJECTOR = {RandomTensorFI.__name__: RandomTensorFI}
