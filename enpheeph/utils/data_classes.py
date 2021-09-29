import dataclasses
import functools
import typing

import enpheeph.utils.enums
import enpheeph.utils.typings


# here are all the info required for injecting faults in a bit
# we need a dataclass so that we can convert the BitFaultValue type into
# a mask with fill values
@dataclasses.dataclass
class BitFaultMaskInfo(object):
    operation: enpheeph.utils.enums.FaultMaskOperation
    mask_value: enpheeph.utils.enums.FaultMaskValue
    fill_value: enpheeph.utils.enums.FaultMaskValue

    @classmethod
    def from_bit_fault_value(
            cls,
            bit_fault_value: enpheeph.utils.enums.BitFaultValue,
    ) -> BitFaultMaskInfo:
        return cls.BIT_FAULT_VALUE_TO_BIT_FAULT_MASK_INFO[bit_fault_value]


# we can safely assume that the dimension will be 1 only, as this is supposed
# to be used internally from a linear array of bits
@dataclasses.dataclass
class BitIndexInfo(object):
    bit_index: enpheeph.utils.typings.BitIndexType
    # we can use an enum if only a set of bitwidths is allowed
    # bitwidth: enpheeph.utils.enums.BitWidth
    bitwidth: int
    # this is equivalent for big endian
    # NOTE: endianness is not required when we are working at Python level
    # this is because all LSBs are positioned at bit 0 when accessing an 
    # integer, while the corresponding string has MSB at 0
    endianness: enpheeph.utils.enums.Endianness = enpheeph.utils.enums.Endianness.MSBAtIndexZero


@dataclasses.dataclass
class FaultLocation(object):
    # location of the fault injection
    injection_location: InjectionLocation
    # value of fault to be injected
    bit_fault_value: enpheeph.utils.enums.BitFaultValue


@dataclasses.dataclass
class InjectionLocation(object):
    # type of parameter, activation or weight
    parameter_type: enpheeph.utils.enums.ParameterType
    # tensor index which can be represented using a numpy/pytorch indexing
    # array
    tensor_index: enpheeph.utils.typings.IndexType
    # same for the bit injection info
    bit_index: enpheeph.utils.typings.BitIndexType


setattr(
        BitFaultMaskInfo,
        "BIT_FAULT_VALUE_TO_BIT_FAULT_MASK_INFO",
        {
                enpheeph.utils.enums.BitFaultValue.
                StuckAtZero: BitFaultMaskInfo(
                        operation=enpheeph.utils.enums.FaultMaskOperation.And,
                        mask_value=enpheeph.utils.enums.FaultMaskValue.Zero,
                        fill_value=enpheeph.utils.enums.FaultMaskValue.One,
                ),
                enpheeph.utils.enums.BitFaultValue.
                StuckAtOne: BitFaultMaskInfo(
                        operation=enpheeph.utils.enums.FaultMaskOperation.Or,
                        mask_value=enpheeph.utils.enums.FaultMaskValue.One,
                        fill_value=enpheeph.utils.enums.FaultMaskValue.Zero,
                ),
                enpheeph.utils.enums.BitFaultValue.
                BitFlip: BitFaultMaskInfo(
                        operation=enpheeph.utils.enums.FaultMaskOperation.Xor,
                        mask_value=enpheeph.utils.enums.FaultMaskValue.One,
                        fill_value=enpheeph.utils.enums.FaultMaskValue.Zero,
                ),
        },
)