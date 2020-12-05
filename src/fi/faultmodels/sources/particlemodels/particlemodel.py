from . import common
from . import particlemodelabc


# use a more specific name for this Model
class ParticleModel(particlemodelabc.ParticleModelABC):
    def __init__(self, environment: common.Environment, ...):
        kwargs = {}
        kwargs.update(environment=environment)

        super().__init__(self, *args, **kwargs)

        self._environment = environment

    def generate_particles(self, n_particles):
