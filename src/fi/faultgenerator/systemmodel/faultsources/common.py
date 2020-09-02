import collections
import enum
import math
import typing

from ..... import utils


# unique forces the values of the enum to be unique
@enum.unique
# ordered enum automatically implements comparisons
class FaultAreaSize(utils.OrderedEnum):
    """This class implements a descriptor for the area size of a fault.
    """

    SQUARED_UM2 = -2
    SQUARED_MM2 = -1
    SQUARED_CM2 = 0
    SQUARED_DM2 = 1
    SQUARED_M2 = 2


# FIXME: quick implementation for a generic shape, in this case circle
class FaultCircleShape(typing.NamedTuple):

    radius: int

    @property
    def area(self):
        return math.pi * self.radius ** 2


# FIXME: improve these declarations with default types and checks subclassing
# typing.NamedTuple
# fault probability is required for the conversion to error made by the System model
# FIXME: implement also particle strength (Mev) to determine the area of impact
FaultSource2D = collections.namedtuple('FaultSource2D', ['x', 'y', 't', 'fault_probability'])
# this fault identifier uses some parameters like affected area that depend on
# the device model, therefore the fault source returns only the position of the
# hit in relative terms
# FaultIdentifier = collections.namedtuple('FaultIdentifier',
#                                          ['position', 'shape', 'area'])
# we return the positions of the faults, given the total area and execution time
FaultIdentifier = collections.namedtuple('FaultIdentifier',
                                         ['positions'])
