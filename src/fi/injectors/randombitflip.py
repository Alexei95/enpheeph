import pathlib
import sys

import torch

# PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
# if str(PROJECT_DIR) not in sys.path:
#     sys.path.append(str(PROJECT_DIR))

from . import basefi
from . import utils

class RandomBitFlipFI(basefi.BaseFI):
    # only the arguments different from the base classes are listed,
    # the others are in args, kwargs
    def __init__(self, n_bit_flips: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._n_bit_flips = n_bit_flips

    def forward(self, x):
        if self.fi_enabled:
            # total = functools.reduce(operator.mul, x.size())
            # covered = math.ceil(self.percentage * total)
            # index = [tuple(torch.randint(low=0, high=index_dim, size=(1, )).item() for index_dim in x.size()) for _ in range(covered)]
            # r = torch.clone(x).detach()  # add requires_grad_(True) for grad
            # for i in index:
            #     r[i] = torch.randn(size=(1, ))
            r = torch.clone(x).detach().flatten()  # add requires_grad_(True) for grad
            perm = torch.randperm(r.numel())  # we could use cuda but loop is in Python
            # FIXME: samplers must go in a different class
            for i in range(self.n_elements_to_inject(r.numel())):
                r[perm[i]] = utils.bit_flip(r[perm[i]], self._n_bit_flips, sampler=self.element_sampler)
            r = r.reshape(x.size())
        else:
            r = x
        return r

    @property
    def n_bit_flips(self):
        return self._n_bit_flips


FAULT_INJECTOR = {RandomBitFlipFI.__name__: RandomBitFlipFI}
