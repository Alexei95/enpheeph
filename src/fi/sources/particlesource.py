from . import common
from . import sourceabc

class ParticleSource(sourceabc.SourceABC):
    def __init__(self, particle_model: common.ParticleModel, sampler):
        super().__init__({'particle_model': particle_model})

        self._particle_model = particle_model  # for altitude and other stats
        self._sampler = sampler

    def _generate_strike(self, chip_area, execution_time):
        n_particles = self._particle_model.flux_intensity * chip_area * execution_time



    # NOTE: choose between single parameters for better configurability or
    # the whole hardware model, which seems a bit overkill for now
    def generate_strikes(self, n_iterations, chip_area, execution_time):
        strike_sets = []
        for i in range(n_iterations):
