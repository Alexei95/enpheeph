import numpy.random

from ...common import DEFAULT_PRNG, DEFAULT_PRNG_SEED

class BaseSampler(object):
    def __init__(self, bit_generator=DEFAULT_PRNG, seed=DEFAULT_PRNG_SEED, *args, **kwargs):
        super().__init__()

        self._seed = seed
        self._seed_sequence = numpy.random.SeedSequence(self._seed)
        self._bg = DEFAULT_PRNG(self._seed_sequence)
        self._prng = numpy.random.Generator(self._bg)

    @property
    def prng(self):
        return self._prng
