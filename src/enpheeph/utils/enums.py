# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2023 Alessio "Alexei95" Colucci
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

import enum
import operator


class BitFaultValue(enum.Enum):
    # Random = enum.auto()
    StuckAtZero = enum.auto()
    StuckAtOne = enum.auto()
    BitFlip = enum.auto()


class BitWidth(enum.IntEnum):
    OneByte = 8
    TwoBytes = 16
    ThreeBytes = 24
    FourBytes = 32
    FiveBytes = 40
    SixBytes = 48
    SevenBytes = 56
    EightBytes = 64

    FloatingPoint16 = TwoBytes
    FloatingPoint32 = FourBytes
    FloatingPoint64 = EightBytes
    Int32 = FourBytes
    Int64 = EightBytes


class DimensionType(enum.Enum):
    BitLevel = enum.auto()
    Batch = enum.auto()
    Tensor = enum.auto()
    Time = enum.auto()


# NOTE: this endianness does not represent the actual endianness of the machine,
# only the endianness seen in the Python objects when accessing them
class Endianness(enum.Enum):
    Little = "<"
    Big = ">"

    MSBAtIndexZero = Big
    LSBAtIndexZero = Little


class FaultMaskOperation(enum.Enum):
    InPlaceXor = operator.ixor
    InPlaceAnd = operator.iand
    InPlaceOr = operator.ior
    Xor = operator.xor
    And = operator.and_
    Or = operator.or_


class FaultMaskValue(enum.IntEnum):
    One = 1
    Zero = 0


class HandlerStatus(enum.Enum):
    Running = enum.auto()
    Idle = enum.auto()


# we use flag so that different metrics can be composed together
class MonitorMetric(enum.Flag):
    StandardDeviation = enum.auto()
    Maximum = enum.auto()
    Minimum = enum.auto()
    ArithmeticMean = enum.auto()
    GeometricMean = enum.auto()


class ParameterType(enum.Flag):
    # network type
    DNN = enum.auto()
    SNN = enum.auto()

    # sub-network type, as we need special care for RNN
    RNN = enum.auto()

    # parameter type
    Weight = enum.auto()
    Activation = enum.auto()
    State = enum.auto()

    # state types
    LIF = enum.auto()

    # variables saved in state
    Voltage = enum.auto()
    Current = enum.auto()

    # tensor type
    Dense = enum.auto()
    PrunedDense = enum.auto()
    Sparse = enum.auto()

    # sparse coordinates type
    COO = enum.auto()
    CSR = enum.auto()

    # sparse coordinates
    Index = enum.auto()
    Value = enum.auto()

    # complex types
    DNNWeightDense = DNN | Weight | Dense
    DNNActivationDense = DNN | Activation | Dense
    SNNLIFStateVoltageDense = SNN | State | LIF | Voltage | Dense
