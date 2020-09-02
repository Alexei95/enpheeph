import enum
import typing

import numpy

from ..... import utils
# FIXME: fix these imports and improve their structure
from ..devices import common


@enum.unique
class FaultTarget(enum.Flag):
    WEIGHTS = common.ChipElementType.MEMORY
    ACTIVATIONS = common.ChipElementType.REGISTER | common.ChipElementType.ALU_FPU | common.ChipElementType.BUFFER | common.ChipElementType.INTERCONNECTIONS

# FIXME: for now we assume it to be bitflip, but we can implement any of them
# in this way, even though it must be moved to devices.common
@enum.unique
class FaultType(enum.Flag):
    BITFLIP = enum.auto()
    STUCK_AT_0 = enum.auto()
    STUCK_AT_1 = enum.auto()
