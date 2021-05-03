import copy
import typing

import src.fi.utils.dataclasses.binaryfaultmask
import src.fi.utils.enums.binaryfaultmaskop
import src.fi.utils.enums.bitvalue
import src.fi.utils.enums.endianness


# this class is the base as it handles the injection on a binary string
class BinaryHandler(object):
    @classmethod
    def inject_fault_single(
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
    def inject_fault_multi(cls,
            binaries: typing.Sequence[str],
            bit_index: typing.Sequence[int],
            endianness: src.fi.utils.enums.endianness.Endianness,
            bit_value: src.fi.utils.enums.bitvalue.BitValue,
    ) -> str:
        injected_binaries = []
        for binary in binaries:
            injected_binaries.append(cls.inject_fault_single(
                    binary=binary,
                    bit_index=bit_index,
                    endianness=endianness,
                    bit_value=bit_value,
            ))
        return injected_binaries

    @classmethod
    def generate_fault_mask(
            cls,
            bit_width: int,
            bit_index: typing.Sequence[int],
            endianness: src.fi.utils.enums.endianness.Endianness,
            bit_value: src.fi.utils.enums.bitvalue.BitValue,
    ) -> src.fi.utils.dataclasses.binaryfaultmask.BinaryFaultMask:
        # if we are using little endian we invert the index, as the LSB is
        # at the end of the list
        if endianness == endianness.Little:
            index = (len(injected_binary) - 1) - index

        if bit_value in (bit_value.StuckAtOne, bit_value.BitFlip):
            binary_mask = '0' * bit_width
            for index in bit_index:
                binary_mask[index] = '1'
            if bit_value == bit_value.StuckAtOne:
                binary_mask_op = src.fi.utils.enums.\
                        binaryfaultmaskop.BinaryFaultMaskOp.OR
            elif bit_value == bit_value.BitFlip:
                binary_mask_op = src.fi.utils.enums.\
                        binaryfaultmaskop.BinaryFaultMaskOp.XOR
        elif bit_value == bit_value.StuckAtZero:
            binary_mask = '1' * bit_width
            for index in bit_index:
                binary_mask[index] = '0'
            binary_mask_op = \
                    src.fi.utils.enums.binaryfaultmaskop.BinaryFaultMaskOp.OR
        else:
            raise ValueError('Unsupported injection type.')
        return src.fi.utils.dataclasses.binaryfaultmask.BinaryFaultMask(
            mask=binary_mask,
            operation=binary_mask_op,
        )
