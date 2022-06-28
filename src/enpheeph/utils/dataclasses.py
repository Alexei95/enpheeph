# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import abc
import dataclasses
import typing

import enpheeph.utils.classes
import enpheeph.utils.enums
import enpheeph.utils.typings


# all the following dataclasses are frozen as their arguments should not change
# this also simplifies the handling of PickleType for the SQL storage plugin
#
# here are all the info required for injecting faults in a bit
# we need a dataclass so that we can convert the BitFaultValue type into
# a mask with fill values
@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
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
        cls,
        bit_fault_value: enpheeph.utils.enums.BitFaultValue,
    ) -> "BitFaultMaskInfo":
        dict_: typing.Dict[
            str, typing.Any
        ] = cls.BIT_FAULT_VALUE_TO_BIT_FAULT_MASK_INFO_ARGS[bit_fault_value]
        return cls(**dict_)


# we can safely assume that the dimension will be 1 only, as this is supposed
# to be used internally from a linear array of bits
@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class BitIndexInfo(object):
    bit_index: enpheeph.utils.typings.Index1DType
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


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class LocationModuleNameMixin(object):
    # name of the module to be targeted
    module_name: str


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class LocationMixin(object):
    # parameter, activation or weight type
    parameter_type: enpheeph.utils.enums.ParameterType
    # same for the bit injection info
    bit_index: enpheeph.utils.typings.Index1DType


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class LocationOptionalMixin(object):
    # name of parameters to get, default is None as it is required if it is not
    # an activation injection
    parameter_name: typing.Optional[str] = None
    # batch/tensor/time indices are now inside the dimension_index
    dimension_index: typing.Optional[
        enpheeph.utils.typings.DimensionLocationIndexType
    ] = None
    # mask for batch/tensor/time
    dimension_mask: typing.Optional[
        enpheeph.utils.typings.DimensionLocationMaskType
    ] = None

    def __post_init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        # not needed, it should be done in sub-classes
        # super().__post_init__(*args, **kwargs)

        not_activation_type = (
            self.parameter_type.Activation  # type: ignore[attr-defined]
            not in self.parameter_type  # type: ignore[attr-defined]
        )
        at_least_one_dimension = (
            self.dimension_index is not None or self.dimension_mask is not None
        )

        if not_activation_type and self.parameter_name is None:
            raise ValueError(
                "'parameter_name' must be provided "
                "if the type of parameter is not an activation"
            )
        if not at_least_one_dimension:
            raise ValueError(
                "at least one between 'dimension_index' and "
                "'dimension_mask' must be given"
            )
        else:
            dim_index = self.dimension_index if self.dimension_index is not None else {}
            dim_mask = self.dimension_mask if self.dimension_mask is not None else {}
            overlap_dimension = set(dim_index.keys()).intersection(dim_mask.keys())
            if overlap_dimension:
                raise ValueError("dimensions overlap some indices")


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class FaultLocationMixin(object):
    # value of fault to be injected
    bit_fault_value: enpheeph.utils.enums.BitFaultValue


# the order of the parameters is from last to first
# so the ones with defaults should be at the beginning
# NOTE: we define post-init to generate the id for each class
# if overriding post-init in the subclasses, call it with super() for id generation
@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class InjectionLocationABC(
    LocationModuleNameMixin,
    enpheeph.utils.classes.IDGenerator,
    abc.ABC,
    object,
    shared_root_flag=True,
):
    pass


# here we define a common base injection location, to use the basic parameters
# which are in common to Monitor and Fault
@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class BaseInjectionLocation(
    LocationMixin,
    InjectionLocationABC,
    use_shared=True,
):
    pass


# the order of the parameters is from last to first
# so the ones with defaults should be at the beginning
@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class MonitorLocation(
    LocationOptionalMixin,
    BaseInjectionLocation,
    use_shared=True,
):
    pass


# the order of the parameters is from last to first
# so the ones with defaults should be at the beginning
@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True, unsafe_hash=True)
class FaultLocation(
    LocationOptionalMixin,
    FaultLocationMixin,
    BaseInjectionLocation,
    use_shared=True,
):
    pass
