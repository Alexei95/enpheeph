# here we implement the model for the fault propagation in the hardware
# the basic idea is to partition the whole hardware chip (GPU, FPGA, ...)
# into small squares
# this matrix (which will be 3D in the future, to allow modeling also the third
# dimension when particles hit the chip) will have each square being associated
# at different levels with the most common element in there, i.e. for phyisical
# level it will be compute/memory/interconnection/...
# this will work at different layers (physical, architectural, software)
# for mapping the faults at multiple levels

import collections
import dataclasses
import enum
import typing


PhysicalType = enum.Enum('PhysicalType', 'Compute Control StaticMemory DynamicMemory Interconnection')


@dataclasses.dataclass(init=True)
class Cell(object):
    # FIXME: other info can be added, like other physical properties
    width: float
    # width_unit: LengthMeasurementUnit -> new enum/dataclass with prefixes for cm/mm/m, ...
    height: float
    #depth: float = None

    #@property
    #def twodimensional(self):
    #    return self.depth is None


@dataclasses.dataclass(init=True)
class HardwareInfo(object):
    # FIXME: other info can be added, like other physical properties
    width: float
    # width_unit: LengthMeasurementUnit -> new enum/dataclass with prefixes for cm/mm/m, ...
    height: float
    # depth: float = None

    #@property
    #def twodimensional(self):
    #    return self.depth is None


@dataclasses.dataclass(init=True)
class ModelMap(object):
    cells: typing.Tuple[typing.Tuple[ArchitecturalType]]

@dataclasses.dataclass(init=True)
class HardwareModel(object):
    hardware_info: HardwareInfo
    model_map: ModelMap
    architectural_map: typing.Tuple[typing.Tuple[ArchitecturalType]]

    def __post_init__(self):
        pass
