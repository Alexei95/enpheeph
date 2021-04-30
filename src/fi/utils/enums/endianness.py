import enum


class Endianness(enum.Flag):
    Little = enum.auto()
    Big = enum.auto()

    MSBAtIndexZero = Big
    LSBAtIndexZero = Little
