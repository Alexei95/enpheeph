import typing

from . import common
from . import faultsourceabc


# FIXME: implement this fixed module as a call to the webpage using requests or
# other POST libraries
# http://www.seutest.com/cgi-bin/FluxCalculator.cgi
class ParticleSource(faultsourceabc.FaultSourceABC):
    def __init__(self, flux, sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._flux = flux
        self._sampler = sampler

    @property
    def flux(self):
        return self._flux

    @property
    def sampler(self):
        return self._sampler

    # we return a list of fault positions, depending on x, y and t
    def generate_fault_sources(self, area, time) -> typing.Tuple[common.FaultSource2D]:
        # we assume area in cm2 and time in hours
        # the number of particles hit depend on area, time and flux
        n_of_neutrons = area * time * self._flux
        res = []
        for neutron in range(n_of_neutrons):
            # all the coordinates are relative
            pos_x = self._sampler()
            pos_y = self._sampler()
            t = self._sampler()
            fault_probability = self._sampler()
            res.append(common.FaultSource2D(pos_x, pos_y, t, fault_probability))
        return tuple(res)
