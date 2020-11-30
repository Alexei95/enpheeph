import enum


class SiliconType(enum.IntFlag):
    REGISTER = enum.auto()
    ALU = enum.auto()
    FPU = enum.auto()
    CACHE = enum.auto()
    MEMORY = enum.auto()
    INTERCONNECTION = enum.auto()
    CONTROL_LOGIC = enum.auto()
