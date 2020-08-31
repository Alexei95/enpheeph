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

    def generate_faults(self, area, time) -> typing.Tuple[common.Position2DplusT]:
        # we assume area in cm2 and time in hours
        n_of_neutrons = area * time * self._flux
        res = []
        for neutron in range(n_of_neutrons):
            # all the coordinates are relative
            pos_x = self._sampler()
            pos_y = self._sampler()
            t = self._sampler()
            res.append(common.Position2DplusT(pos_x, pos_y, t))
        return tuple(res)
