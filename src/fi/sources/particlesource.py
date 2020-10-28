import typing

from ..faultmodels import common as fault_model_common
from . import common
from . import sourceabc

class ParticleSource(sourceabc.SourceABC):
    def __init__(self, particle_model: common.ParticleModel, sampler):
        super().__init__({'particle_model': particle_model})

        self._particle_model = particle_model  # for altitude and other stats
        self._sampler = sampler

    def _generate_strike(self, hardware_info: fault_model_common.HardwareInfo):
        strikes = []
        n_particles = self._particle_model.flux_intensity * hardware_info.chip_area * hardware_info.execution_time
        for i in range(n_particles):
            energy = self._particle_model.process_relative_energy(self._sampler.sample())
            # + 1 is for time, the others are space
            rel_space_time_coordinates = self._sampler.sample(n_samples=len(hardware_info.chip_dimensions) + 1)
            abs_space_time_coordinates = hardware_info.process_relative_coordinates(rel_space_time_coordinates)
            strikes.append(common.Particle(coordinates=abs_space_time_coordinates, energy=energy))
        return strikes

    # NOTE: choose between single parameters for better configurability or
    # the whole hardware model, which seems a bit overkill for now
    def generate_strikes(self, n_iterations, hardware_info: fault_model_common.HardwareInfo) -> list[list[common.Particle]]:
        strike_sets = []
        for i in range(n_iterations):
            strike_sets.append(self._generate_strike(hardware_info=hardware_info))

