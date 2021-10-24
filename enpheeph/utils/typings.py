# -*- coding: utf-8 -*-
import pathlib
import typing

# we use Tuple and not Sequence to allow hashability
BitIndexType = typing.TypeVar("BitIndexType",
    int,
    slice,
    # NOTE: in Python 3.10 there is types.EllipsisType
    type(Ellipsis),
    typing.Sequence[int,],
)
IndexType = typing.TypeVar("IndexType",
    int,
    slice,
    type(Ellipsis),
    typing.Sequence[typing.Union[int, slice, type(Ellipsis), typing.Tuple[int,],],],
)
TimeIndexType = typing.TypeVar("TimeIndexType",
    int,
    slice,
    # NOTE: in Python 3.10 there is types.EllipsisType
    type(Ellipsis),
    typing.Sequence[int,],
)

PathType = typing.TypeVar("PathType",
    str,
    pathlib.Path
)

LowLevelMaskArrayType = typing.TypeVar(
    "LowLevelMaskArrayType", "cupy.ndarray", "numpy.ndarray",
)
ModelType = typing.TypeVar("ModelType", "torch.nn.Module", typing.Any,)
