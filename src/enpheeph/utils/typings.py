# -*- coding: utf-8 -*-
import pathlib
import typing

# we fake import cupy, numpy and torch to silence mypy
if typing.TYPE_CHECKING:
    # invalid input for pyflakes
    from builtins import ellipsis

    import cupy
    import numpy
    import torch

import enpheeph.utils.enums

# for the active_dimension_index
ActiveDimensionIndexType = typing.Union[
    enpheeph.utils.enums.DimensionType,
    # NOTE: in Python 3.10 there is types.EllipsisType
    # if we skip the following type mypy goes crazy
    # however a fix is to use "ellipsis", and add builtins.ellipsis at the top
    "ellipsis",
]

# we could even add bit and other parameters in here
AnyIndexType = typing.Union[
    "Index1DType",
    "IndexMultiDType",
]
DimensionDictType = typing.Dict[
    enpheeph.utils.enums.DimensionType,
    "DimensionIndexType",
]
DimensionIndexType = typing.Union[
    int,
    # NOTE: in Python 3.10 there is types.EllipsisType
    # if we skip the following type mypy goes crazy
    # however a fix is to use "ellipsis", and add builtins.ellipsis at the top
    "ellipsis",
    # **NOTE**: we do not support tuples yet, one can duplicate enum values to have
    # multiple indices with similar names
    # typing.Tuple[int, ...],
]
DimensionLocationIndexType = typing.Dict[
    enpheeph.utils.enums.DimensionType,
    AnyIndexType,
]
# we use Tuple and not Sequence to allow hashability
# mypy reports error if one of the types is not valid
Index1DType = typing.Union[
    int,
    slice,
    # NOTE: in Python 3.10 there is types.EllipsisType
    # if we skip the following type mypy goes crazy
    # however a fix is to use "ellipsis", and add builtins.ellipsis at the top
    "ellipsis",
    # we need List as Tuple is seen as multiple dimensions when indexing
    # **NOTE**: this might give problems with hashing in the dataclasses
    typing.List[int],
]
IndexMultiDType = typing.Union[
    int,
    slice,
    # NOTE: in Python 3.10 there is types.EllipsisType
    # if we skip the following type mypy goes crazy
    # however a fix is to use "ellipsis", and add builtins.ellipsis at the top
    "ellipsis",
    # we use Tuple as in this case we need to cover multiple dimensions
    typing.Tuple[Index1DType, ...],
]
IndexTimeType = Index1DType

PathType = typing.Union[str, pathlib.Path]

LowLevelMaskArrayType = typing.Union[
    "cupy.ndarray",
    "numpy.ndarray",
]
ModelType = typing.Union[
    "torch.nn.Module",
    typing.Any,
]

ShapeType = typing.Tuple[int, ...]

TensorType = typing.Union[
    "cupy.ndarray",
    "numpy.ndarray",
    "torch.Tensor",
]
