import abc

import numpy.random
import torch

from ...common import DEFAULT_PRNG, DEFAULT_PRNG_SEED


class BaseSampler(object):
    def __init__(self, seed=DEFAULT_PRNG_SEED, bit_generator=DEFAULT_PRNG, *args, **kwargs):
        super().__init__()

        self._seed = seed
        self._seed_sequence = numpy.random.SeedSequence(self._seed)
        self._bg = DEFAULT_PRNG(self._seed_sequence)
        self._prng = numpy.random.Generator(self._bg)

    @property
    def prng(self):
        return self._prng

    # this set of arguments is the standard ones, but they could be different
    # so we use also *args, **kwargs
    @abc.abstractmethod
    def iter_choice(self, high=None, n_elements=None, low=None, unique=None, *args, **kwargs):
        return iter([])

    # this set of arguments is the standard ones, but they could be different
    # so we use also *args, **kwargs
    @abc.abstractmethod
    def torch_sample(self, high=None, shape=None, low=None, dtype=None, device=None, *args, **kwargs):
        return torch.zeros(())
