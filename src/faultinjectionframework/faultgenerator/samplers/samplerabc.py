import abc

import numpy.random

from ....common import DEFAULT_PRNG, DEFAULT_PRNG_SEED
from .common import DEFAULT_SAMPLER_INDEX_CLASS


class SamplerABC(abc.ABC):
    # we use this class attribute as default SamplerIndex class
    # it should support tensor_index and bit_index, to allow usage with
    # all possible operations
    _index_class = DEFAULT_SAMPLER_INDEX_CLASS

    def __init__(self, seed=DEFAULT_PRNG_SEED, bit_generator=DEFAULT_PRNG, *args, **kwargs):
        super().__init__()

        self._seed = seed
        self._seed_sequence = numpy.random.SeedSequence(self._seed)
        self._bit_generator = DEFAULT_PRNG(self._seed_sequence)
        self._prng = numpy.random.Generator(self._bit_generator)

    @property
    def prng(self):
        return self._prng

    def make_index(self, *args, **kwargs):
        return self._index_class(*args, **kwargs)

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
