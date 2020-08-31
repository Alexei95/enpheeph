import numpy.random

from ....common import DEFAULT_PRNG
from . import samplerabc


class NumpySampler(samplerabc.SamplerABC):
    def __init__(self, bit_generator=DEFAULT_PRNG, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._seed_sequence = numpy.random.SeedSequence(self._seed)
        self._bit_generator = DEFAULT_PRNG(self._seed_sequence)
        self._prng = numpy.random.Generator(self._bit_generator)

    @property
    def prng(self):
        return self._prng

    @property
    def bit_generator(self):
        return self._bit_generator

    def __call__(self, *args, **kwargs):
        # FIXME: missing implementation, it depends on the source
        pass
