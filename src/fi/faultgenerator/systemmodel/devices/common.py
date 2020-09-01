import enum
import typing

import numpy

from ..... import utils


# unique forces the values of the enum to be unique
@enum.unique
# ordered enum automatically implements comparisons
class ChipElementSizeUnit(utils.OrderedEnum):
    """This class implements a descriptor for the unit used for the minimum area
    size when dividing a chip into sub-elements.
    """

    SQUARED_UM2 = -12
    SQUARED_MM2 = -6
    SQUARED_CM2 = -4
    SQUARED_DM2 = -2
    SQUARED_M2 = 0


@enum.unique
class ChipElementType(enum.Flag):
    REGISTER = enum.auto()
    CONTROL = enum.auto()
    MEMORY = enum.auto()
    ALU_FPU = enum.auto()
    BUFFER = enum.auto()
    INTERCONNECTIONS = enum.auto()


