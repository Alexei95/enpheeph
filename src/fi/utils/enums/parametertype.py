import enum


class ParameterType(enum.Flag):
    Weight = enum.auto()
    Activation = enum.auto()
    Sparse = enum.auto()
    COO = enum.auto()
    Index = enum.auto()
    Value = enum.auto()
    SparseWeightCOOIndex = Sparse | Weight | COO | Index
    SparseWeightCOOValue = Sparse | Weight | COO | Value
    SparseActivationCOOIndex = Sparse | Activation | COO | Index
    SparseActivationCOOValue = Sparse | Activation | COO | Value
