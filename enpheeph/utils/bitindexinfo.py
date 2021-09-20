import dataclasses
import typing

import enpheeph.utils.typings


# we can safely assume that the dimension will be 1 only, as this is supposed
# to be used internally from a linear array of bits
@dataclasses.dataclass
class BitIndexInfo(object):
    bit_index: enpheeph.utils.typings.IndexType
    bitwidth: int
    endianness: enpheeph.utils.enums.Endianness
