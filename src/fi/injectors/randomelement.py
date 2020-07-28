import math
import pathlib
import sys

import torch

# PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
# if str(PROJECT_DIR) not in sys.path:
#     sys.path.append(str(PROJECT_DIR))

from .import basefi

class RandomElementFI(basefi.BaseFI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.fi_enabled:
            # total = functools.reduce(operator.mul, x.size())
            # covered = math.ceil(self.percentage * total)
            # index = [tuple(torch.randint(low=0, high=index_dim, size=(1, )).item() for index_dim in x.size()) for _ in range(covered)]
            # r = torch.clone(x).detach()  # add requires_grad_(True) for grad
            # for i in index:
            #     r[i] = torch.randn(size=(1, ))
            r = torch.clone(x).detach().flatten()  # add requires_grad_(True) for grad
            for i in self.index_to_inject(r.numel()):
                r[i] = self.element_sampler.torch_sample(low=0, high=1, size=(1, ), dtype=x.dtype, device=x.device)
            r = r.reshape(x.size())
        else:
            r = x
        return r

FAULT_INJECTOR = {RandomElementFI.__name__: RandomElementFI}
