import copy
import typing

import src.fi.utils.enums.bitvalue
import src.fi.utils.enums.endianness


# this class is the base as it handles the injection on a binary string
class BinaryHandler(object):
    @classmethod
    def inject_fault(
            cls,
            binary: str,
            bit_index: typing.Sequence[int],
            endianness: src.fi.utils.enums.endianness.Endianness,
            bit_value: src.fi.utils.enums.bitvalue.BitValue,
    ) -> str:
        # we need to convert the binary string into a list of characters
        # otherwise we cannot update the values
        injected_binary = list(copy.deepcopy(binary))
        for index in bit_index:
            # if we are using little endian we invert the index, as the LSB is
            # at the end of the list
            if endianness == endianness.Little:
                index = (len(injected_binary) - 1) - index

            if bit_value == bit_value.StuckAtOne:
                injected_binary[index] = "1"
            elif bit_value == bit_value.StuckAtZero:
                injected_binary[index] = "0"
            elif bit_value == bit_value.BitFlip:
                injected_binary[index] = str(int(injected_binary[index]) ^ 1)
            else:
                raise ValueError('Unsupported injection type.')
        return ''.join(injected_binary)

    @classmethod
    def generate_fault_mask(cls):
        pass
