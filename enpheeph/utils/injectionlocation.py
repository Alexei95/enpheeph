import dataclasses
import typing

import enpheeph.utils.bitindexinfo
import enpheeph.utils.typings


@dataclasses.dataclass
class InjectionLocation(object):
    # type of parameter, activation or weight
    parameter_type: enpheeph.utils.enums.ParameterType
    # tensor index which can be represented using a numpy/pytorch indexing
    # array
    tensor_index: enpheeph.utils.typings.IndexType
    # same for the bit injection info
    bit_index_info: enpheeph.fault.bitindexinfo.BitIndexInfo
    # value of fault to be injected
    bit_fault_value: enpheeph.utils.enums.BitFaultValue
