import abc
import collections
import math

import torch

class BaseFI(torch.nn.Module):
    # FIX: this is a __init__ function, but the issue is when there is multiple
    # inheritance regarding the arguments to be passed, so it was renamed as
    # init must be called before executing the FI (ideally in __init__)
    # FIX2: since now BaseFI inherits from Module we can call this __init__
    def __init__(self, coverage=1.0, n_elements=None, enableFI=True, *args, **kwargs):
        super().__init__()
        self._enableFI = enableFI
        if (coverage is None and n_elements is None) or (coverage is not None and n_elements is not None):
            # FIXME: improve Exception type
            raise Exception('Only one of coverage and n_elements must be different from None')
        self._coverage = coverage
        self._n_elements = n_elements

    # forward must be implemented, in any case this is just a bypass
    @abc.abstractmethod
    def forward(self, x):
        return x

    def turn_on_fi(self):
        self._enableFI = True

    def turn_off_fi(self):
        self._enableFI = False

    def toggle_fi(self):
        self._enableFI = not self._enableFI

    @property
    def fi_enabled(self):
        return self._enableFI

    def n_elements_to_inject(self, total_size):
        if self._n_elements is not None:
            return min(max(0, self._n_elements), total_size)
        elif self._coverage is not None:
            # we ceil the result of the percentage coverage
            return min(max(0, math.ceil(self._coverage * total_size)), total_size)
        else:
            # FIXME: improve Exception type
            raise Exception('Only one of coverage and n_elements must be different from None')

    @staticmethod
    def setup_fi(fi_obj, module):
        return torch.nn.Sequential(collections.OrderedDict([
                        ('original', module),
                        ('fi', fi_obj)]))
