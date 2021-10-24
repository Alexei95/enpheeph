# -*- coding: utf-8 -*-
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


# this endianness does not represent the actual endianness of the machine,
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
    LIFState = enum.auto()

    # variables saved in state
    Voltage = enum.auto()
    Current = enum.auto()

    # tensor type
    Dense = enum.auto()
    Sparse = enum.auto()

    # sparse coordinates type
    COO = enum.auto()

    # sparse coordinates
    Index = enum.auto()
    Value = enum.auto()

    # complex types
    DNNWeightDense = DNN | Weight | Dense
    DNNActivationDense = DNN | Activation | Dense
    SNNStateLIFStateVoltageDense = SNN | State | LIFState | Voltage | Dense
