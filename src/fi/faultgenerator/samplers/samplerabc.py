import abc

from ....common import DEFAULT_PRNG, DEFAULT_PRNG_SEED


class SamplerABC(abc.ABC):
    def __init__(self, seed=DEFAULT_PRNG_SEED, *args, **kwargs):
        super().__init__()

        self._seed = seed

    @property
    def seed(self):
        return self._seed

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """This abstract method must be implemented to sample a distribution.

        It must return a value between 0 and 1."""
        pass
