import enum


class FaultMaskOp(enum.Enum):
    XOR = enum.auto()
    AND = enum.auto()
    OR = enum.auto()
