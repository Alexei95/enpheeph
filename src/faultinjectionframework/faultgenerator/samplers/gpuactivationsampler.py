import abc

import numpy.random
import torch


class GPUActivationSampler(abc.ABC):
    def __init__(self, gpu_distribution, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._gpu_distribution = gpu_distribution

    @property
    def prng(self):
        return self._prng

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


SAMPLER = {GPUActivationSampler.__name__: GPUActivationSampler}
