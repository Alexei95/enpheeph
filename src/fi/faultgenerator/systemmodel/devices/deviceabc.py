import abc
import typing

import numpy

from . import common
from ..faultsources.common import FaultSource2D
from ..samplers import samplerabc


class DeviceABC(abc.ABC):
    def __init__(self, hit_to_fault_conversion_rate: float,
                 chip_division: typing.Iterable[common.ChipElementType],
                 unit: common.ChipElementSizeUnit,
                 sampler: samplerabc.SamplerABC,
                 *args, **kwargs):

        # FIXME: make this hit to fault conversion rate into a map
        self._hit_to_fault_conversion_rate = hit_to_fault_conversion_rate
        self._chip_division = numpy.array(chip_division)
        self._unit = unit
        self._sampler = sampler

        self.post_init()

    def post_init(self):
        self._asserts()

        # we pass the dimensions to a dict so that we can map the different
        # sizes and use a default value
        dims = {i: self._chip_division.shape[i] for i in range(self._chip_division.shape)}

        self._x = self._chip_division.shape.get(0)
        self._y = self._chip_division.shape.get(1)
        self._z = self._chip_division.shape.get(2)

    def _asserts(self):
        assert all(isinstance(x, common.ChipElementType) for x in self._chip_division.reshape(-1))

    # FIXME: implement a custom number of coordinates to convert
    # FIXME: use SI units
    def convert_relative_coordinates_to_indices(self, position_x, position_y):
        # round works following half even
        abs_pos_x = round(position_x * self._x)
        abs_pos_y = round(position_y * self._y)
        return (abs_pos_x, abs_pos_y)

    # FIXME: use particle energy to determine size of impact
    # for now only the positioning in the squares determines the hit or miss
    def convert_sources_to_faults(self, sources: typing.Tuple[FaultSource2D]):
        faults = []
        for source in sources:
            probability = source.fault_probability

            # we check the probability of the fault with the conversion
            # rate, if higher we skip it
            if probability > self.hit_to_fault_conversion_rate:
                continue

            # we convert the position to an index
            x, y = source.x, source.y
            index_x, index_y = self.convert_relative_coordinates_to_indices(x, y)

            faulty_area = self._chip_division[index_x, index_y]

            # FIXME: convert this list to a namedtuple
            faults.append([faulty_area, source.t])

        return faults

    @property
    @abc.abstractmethod
    def execution_order(self):
        pass

    @abc.abstractmethod
    def execution_time(self, *args, **kwargs):
        pass

    @property
    def hit_to_fault_conversion_rate(self):
        return self._hit_to_fault_conversion_rate
