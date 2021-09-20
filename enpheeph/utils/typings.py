import typing

# we use Tuple and not Sequence to allow hashability
BitIndexType = typing.Union[
        int,
        slice,
        Ellipsis,
        typing.Tuple[
                int,
                ...
        ]
]
IndexType = typing.Union[
        int,
        slice,
        Ellipsis,
        typing.Tuple[
                typing.Union[
                        int,
                        slice,
                        Ellipsis,
                        typing.Tuple[
                                int,
                                ...
                        ]
                ]
        ]
]

# module type for generic injection
ModuleType = typing.TypeVar("ModuleType")
