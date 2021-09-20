import enum


class BitFaultValue(enum.Enum):
    Random = enum.auto()
    StuckAtZero = enum.auto()
    StuckAtOne = enum.auto()
    BitFlip = enum.auto()


class Endianness(enum.Flag):
    Little = enum.auto()
    Big = enum.auto()

    MSBAtIndexZero = Big
    LSBAtIndexZero = Little


class FaultMaskOp(enum.Enum):
    XOR = enum.auto()
    AND = enum.auto()
    OR = enum.auto()


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
