import dataclasses

import math


@dataclasses.dataclass
class ParticleModel:
    #latitude: float  # not needed for now
    #longitude: float  # not needed for now
    #altitude: float  # not needed for now
    flux_intensity: float

    # here I am assuming a Gaussian distribution, but it can be changed by
    # subclassing
    # basically is sort of Gaussian for altitude
    # exponentially decreasing for flux (energy = exp(-flux))
    energy_average: float
    energy_std: float

    def process_relative_energy(self, sample, *args, **kwargs):
        # gaussian assumption for energy
        return math.exp(- (sample - self.energy_average) ** 2/ self.energy_std ** 2)
