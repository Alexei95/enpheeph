import numpy
import numpy.random
import torch

from . import basesampler
from ...common import DEFAULT_TORCH_DEVICE, DEFAULT_TORCH_DTYPE

class UniformSampler(basesampler.BaseSampler):
    # seed and prng are set by the base class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # starts from low and reaches high - 1
    # it takes the first n_elements elements on a 1D array
    # if unique is true it avoids placing back in the pool the extracted numbers
    # we return a iterator
    def iter_choice(self, high, n_elements=1, low=0, unique=True, *args, **kwargs):
        perm = self.prng.choice(numpy.arange(low, high), (n_elements, ),
                                replace=not unique)
        return iter(perm)

    def torch_sample(self, high, shape, low=0, dtype=DEFAULT_TORCH_DTYPE, device=DEFAULT_TORCH_DEVICE, *args, **kwargs):
        np = self.prng.uniform(low=low, high=high, size=shape)
        return torch.tensor(np, dtype=dtype, device=device)


SAMPLER = {UniformSampler.__name__: UniformSampler}
