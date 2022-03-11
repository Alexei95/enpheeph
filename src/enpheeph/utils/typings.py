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
AnyMaskType = typing.Union[
    "Mask1DType",
    "MaskMultiDType",
]

ArrayType = typing.Union[
    "cupy.ndarray",
    "numpy.ndarray",
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
DimensionLocationMaskType = typing.Dict[
    enpheeph.utils.enums.DimensionType,
    AnyMaskType,
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
Mask1DType = typing.Sequence[bool]
MaskMultiDType = typing.Union[
    Mask1DType,
    typing.Sequence[Mask1DType],
]

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
    ArrayType,
    "torch.Tensor",
]
