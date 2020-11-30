import dataclasses
import functools
import itertools
import operator
import typing


@dataclasses.dataclass
class HardwareInfo:
    # TODO: use proper containers for numbers with units, e.g. forallpeople
    chip_dimensions: typing.List[float]  # chip dimensions in cm

    execution_time: float  # execution time in microseconds

    @proeprty
    def chip_area(self) -> float:
        # we multiply only the first two dimensions, third is thickness
        return functools.reduce(operator.mul, self.chip_dimensions[:2], 1)

    # support coordinates for space and time, so coordinates is a 4-element list
    # first 3 options are length, width and height
    def process_relative_coordinates(self, coordinates: typing.List[float]):
        abs_dimensions = [*self.chip_dimensions, self.execution_time]
        abs_coordinates = itertools.starmap(zip(coordinates, abs_dimensions))
        return abs_coordinates
