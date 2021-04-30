import enum


class BitValue(enum.Enum):
    Random = enum.auto()
    StuckAtZero = enum.auto()
    StuckAtOne = enum.auto()
    BitFlip = enum.auto()
