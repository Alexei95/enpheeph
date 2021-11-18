# -*- coding: utf-8 -*-
import pathlib
import typing

# we fake import cupy, numpy and torch to silence mypy
if typing.TYPE_CHECKING:
    import cupy  # noqa: F401
    import numpy  # noqa: F401
    import torch  # noqa: F401

# we use Tuple and not Sequence to allow hashability
# mypy reports error if one of the types is not valid
Index1DType = typing.Union[
    int,
    slice,
    # NOTE: in Python 3.10 there is types.EllipsisType
    # if we skip the following type mypy goes crazy
    # type(Ellipsis),  # type: ignore[valid-type]
    typing.Tuple[int, ...],
]
IndexMultiDType = typing.Union[
    int,
    slice,
    # if we skip the following type mypy goes crazy
    # type(Ellipsis),  # type: ignore[valid-type]
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
