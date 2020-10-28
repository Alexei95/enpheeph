import abc


class ParticleModelABC(object):
    def __init__(self, sampler):
        self._sampler = sampler

    @abc.abstractmethod
    def generate_particles(self, n_particles):
        pass
