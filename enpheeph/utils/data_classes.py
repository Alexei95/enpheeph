# -*- coding: utf-8 -*-
import dataclasses
import typing

import enpheeph.utils.classes
import enpheeph.utils.enums
import enpheeph.utils.typings


# here are all the info required for injecting faults in a bit
# we need a dataclass so that we can convert the BitFaultValue type into
# a mask with fill values
@dataclasses.dataclass
class BitFaultMaskInfo(object):
    # to convert bit faults into arguments for the fault mask
    BIT_FAULT_VALUE_TO_BIT_FAULT_MASK_INFO_ARGS = {
        enpheeph.utils.enums.BitFaultValue.StuckAtZero: {
            "operation": enpheeph.utils.enums.FaultMaskOperation.And,
            "mask_value": enpheeph.utils.enums.FaultMaskValue.Zero,
            "fill_value": enpheeph.utils.enums.FaultMaskValue.One,
        },
        enpheeph.utils.enums.BitFaultValue.StuckAtOne: {
            "operation": enpheeph.utils.enums.FaultMaskOperation.Or,
            "mask_value": enpheeph.utils.enums.FaultMaskValue.One,
            "fill_value": enpheeph.utils.enums.FaultMaskValue.Zero,
        },
        enpheeph.utils.enums.BitFaultValue.BitFlip: {
            "operation": enpheeph.utils.enums.FaultMaskOperation.Xor,
            "mask_value": enpheeph.utils.enums.FaultMaskValue.One,
            "fill_value": enpheeph.utils.enums.FaultMaskValue.Zero,
        },
    }

    operation: enpheeph.utils.enums.FaultMaskOperation
    mask_value: enpheeph.utils.enums.FaultMaskValue
    fill_value: enpheeph.utils.enums.FaultMaskValue

    @classmethod
    def from_bit_fault_value(
        cls, bit_fault_value: enpheeph.utils.enums.BitFaultValue,
    ) -> "BitFaultMaskInfo":
        dict_: typing.Dict[str, typing.Any] = cls.BIT_FAULT_VALUE_TO_BIT_FAULT_MASK_INFO_ARGS[bit_fault_value]
        return cls(**dict_)


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
    endianness: enpheeph.utils.enums.Endianness = (
        enpheeph.utils.enums.Endianness.MSBAtIndexZero
    )


@dataclasses.dataclass
class InjectionLocationNoTimeMixIn(object):
    # name of the module to be targeted
    module_name: str
    # type of parameter, activation or weight
    parameter_type: enpheeph.utils.enums.ParameterType
    # tensor index which can be represented using a numpy/pytorch indexing
    # array
    tensor_index: enpheeph.utils.typings.IndexType
    # same for the bit injection info
    bit_index: enpheeph.utils.typings.BitIndexType


@dataclasses.dataclass
class InjectionLocationTimeMixIn(object):
    # index used for time, optional as it is required only for SNNs
    # NOTE: this solution limits the expressivity of InjectionLocation, as we
    # cannot have different injections at different time-steps
    # however, even supporting different masks at different time-steps would
    # still require the creation of multiple masks, as they are created based
    # on the output at each time-step, hence nullifying the actual memory gains
    # the only overhead is the different hooks as well as the multiple Python
    # objects
    time_index: (
        typing.Optional[enpheeph.utils.typings.TimeIndexType]
    ) = dataclasses.field(default=None)


@dataclasses.dataclass
class FaultLocationMixIn(object):
    # value of fault to be injected
    bit_fault_value: enpheeph.utils.enums.BitFaultValue


# the order of the parameters is from last to first
# so the ones with defaults should be at the beginning
@dataclasses.dataclass
class InjectionLocation(InjectionLocationTimeMixIn, InjectionLocationNoTimeMixIn):
    pass


# the order of the parameters is from last to first
# so the ones with defaults should be at the beginning
@dataclasses.dataclass
class FaultLocation(
    InjectionLocationTimeMixIn, InjectionLocationNoTimeMixIn, FaultLocationMixIn,
):
    pass
