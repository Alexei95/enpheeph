import dataclasses
import typing


@dataclasses.dataclass(init=True, repr=True)
class Position(object):
    relative_x: float
    relative_y: float


@dataclasses.dataclass(init=True, repr=True)
class HardwareFault(object):
    position: Position
