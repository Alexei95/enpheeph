import enum


class ParameterType(enum.Flag):
    # network type
    DNN = enum.auto()
    SNN = enum.auto()
    # parameter type
    Weight = enum.auto()
    Activation = enum.auto()
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
