import dataclasses
import typing

import enpheeph.utils.injectionlocation
import enpheeph.utils.typings


@dataclasses.dataclass
class FaultLocation(object):
    # location of the fault injection
    injection_location: enpheeph.utils.injectionlocation.InjectionLocation
    # value of fault to be injected
    bit_fault_value: enpheeph.utils.enums.BitFaultValue
