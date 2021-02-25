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

from . import basefaultdescriptor


PhysicalType = enum.IntFlag('PhysicalType',
                            ['Compute',
                             'Control',
                             'StaticMemory',
                             'DynamicMemory',
                             'Interconnection',
                             ],
                            module=__name__)
ArchitecturalType = enum.IntFlag('ArchitecturalType',
                                 ['',
                                  '',
                                  '',
                                  '',
                                  '',
                                  ],
                                 module=__name__)
SoftwareType = enum.IntFlag('SoftwareType',
                            ['',
                             '',
                             '',
                             '',
                             '',
                             ],
                            module=__name__)

# we use these functions to convert the effects of a fault from a level to
# another
# for now the model is statistical, taken from other papers
def physical_to_architectural(physical_type: PhysicalType,
                              architectural_type: ArchitecturalType):
    pass


@dataclasses.dataclass(init=True)
class HardwareModel(object):
    chip_width: float
    chip_height: float
    chip_measurement_unit: str = "mm^2"

    physical_cells: typing.Tuple[typing.Tuple[PhysicalType]]
    architectural_cells: typing.Tuple[typing.Tuple[ArchitecturalType]]
    software_cells: typing.Tuple[typing.Tuple[SoftwareType]]

    def __post_init__(self):
        pass
