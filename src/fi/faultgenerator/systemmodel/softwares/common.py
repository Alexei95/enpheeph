import enum
import typing

import numpy

from ..... import utils
# FIXME: fix these imports and improve their structure
from ..devices import common


@enum.unique
class FaultType(enum.Flag):
    WEIGHTS = common.ChipElementType.MEMORY
    ACTIVATIONS = common.ChipElementType.REGISTER | common.ChipElementType.ALU_FPU | common.ChipElementType.BUFFER | common.ChipElementType.INTERCONNECTIONS
