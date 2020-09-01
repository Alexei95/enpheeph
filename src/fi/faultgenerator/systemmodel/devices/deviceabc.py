import abc
import typing

import numpy

from . import common
from ..samplers import samplerabc


class DeviceABC(abc.ABC):
    def __init__(self, fault_to_error_conversion_rate: float,
                 chip_division: typing.Iterable[common.ChipElementType],
                 unit: common.ChipElementSizeUnit,
                 sampler: samplerabc.SamplerABC,
                 *args, **kwargs):

        self._fault_to_error_conversion_rate = fault_to_error_conversion_rate
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
        assert all(isinstance(x) for x in self._chip_division.reshape(-1))

    def convert_faults_to_errors(self, faults: )

    @abc.abstractmethod
    def execution_time(self, n_operations):

    @property
    def fault_to_error_conversion_rate(self):
        return self._fault_to_error_conversion_rate

    
