import typing

# we use Tuple and not Sequence to allow hashability
BitIndexType =typing.Union[
        int,
        slice,
        # NOTE: in Python 3.10 there is types.EllipsisType
        type(Ellipsis),
        typing.Sequence[
                int,
                ...
        ]
]
IndexType = typing.Union[
        int,
        slice,
        type(Ellipsis),
        typing.Sequence[
                typing.Union[
                        int,
                        slice,
                        type(Ellipsis),
                        typing.Tuple[
                                int,
                                ...
                        ]
                ],
                ...
        ]
]
LowLevelMaskArrayType = typing.TypeVar(
        "LowLevelMaskArrayType",
        "cupy.ndarray",
        "numpy.ndarray",
)
ModelType = typing.TypeVar(
        "ModelType",
        "torch.nn.Module",
)

