import copy
import functools
import operator
import typing

import numpy


class NumpyConverter(object):
    # uint to avoid double sign repetition
    NUMPY_DATA_CONVERSION_MAPPING = {
            numpy.dtype('float16'): numpy.uint16,
            numpy.dtype('float32'): numpy.uint32,
            numpy.dtype('float64'): numpy.uint64,
            numpy.dtype('uint8'): numpy.uint8,
            numpy.dtype('int8'): numpy.uint8,
            numpy.dtype('int16'): numpy.uint16,
            numpy.dtype('int32'): numpy.uint32,
            numpy.dtype('int64'): numpy.uint64,
    }
    NUMPY_DATA_WIDTH_MAPPING = {
            numpy.dtype('float16'): '16',
            numpy.dtype('float32'): '32',
            numpy.dtype('float64'): '64',
            numpy.dtype('uint8'): '8',
            numpy.dtype('int8'): '8',
            numpy.dtype('int16'): '16',
            numpy.dtype('int32'): '32',
            numpy.dtype('int64'): '64',
    }
    # this template first requires the width (the single {}) and then it can
    # convert a number to a binary view using that width and filling the extra
    # on the left with 0s
    NUMPY_TEMPLATE_STRING = '{{:0{}b}}'

    @classmethod
    def single_numpy_to_binary(cls, element: numpy.ndarray) -> str:
        # if we have different than 1 element, we raise ValueError
        if element.size != 1:
            # FIXME: check which error to raise
            raise ValueError('There must be only 1 element in the array')

        # we remove the extra dimensions using squeeze
        element = element.squeeze()
        # we get the dtype of the element
        dtype = element.dtype
        # we convert data type
        new_dtype = cls.NUMPY_DATA_CONVERSION_MAPPING[dtype]
        # we need the width of the new data type
        width = cls.NUMPY_DATA_WIDTH_MAPPING[dtype]
        # we view the number with a different datatype (int) so we can extract
        # the bits
        str_bin_value = cls.NUMPY_TEMPLATE_STRING.format(
                width
        ).format(
                element.view(
                        new_dtype
                )
        )

        return str_bin_value

    # the outcome will be flattened, to be compatible with Python lists
    @classmethod
    def numpy_to_binary(cls, element: numpy.ndarray) -> typing.List[str]:
        binaries = []
        for el in element.flatten():
            binaries.append(cls.single_numpy_to_binary(el))
        return binaries

    # we need the numpy dtype to return the data in
    # the returned array has () shape, it is a single element
    @classmethod
    def binary_to_numpy(
            cls,
            binary: str,
            dtype: numpy.dtype,
            # for compatibility with cupy interface
            device: typing.Any = None,
    ) -> numpy.ndarray:
        # we need the converted data type, to convert it back to the original
        conv_dtype = cls.NUMPY_DATA_CONVERSION_MAPPING[dtype]

        # we convert the bits to numpy integer through Python int for base 2
        # conversion
        # then we view it back in the original type and convert it to PyTorch
        # square brackets are for creating a numpy.ndarray for PyTorch
        python_int = int(binary, base=2)
        new_numpy_value = conv_dtype(python_int).view(dtype)

        return new_numpy_value

    # this method replicates the same binary value across a set of indices,
    # given also the overall tensor shape
    # fill_value is used to fill the non-selected values in the tensor
    @classmethod
    def binary_to_numpy_broadcast(
            cls,
            binary: str,
            dtype: numpy.dtype,
            index: typing.Sequence[typing.Union[int, slice]],
            shape: typing.Sequence[int],
            fill_value: numpy.ndarray = numpy.array(0),
            # device is required to match the same interface as cupy
            device: typing.Any = None,
    ):
        # we convert the fill_value to a numpy array with the correct dtype
        fill_value = numpy.array(fill_value, dtype=dtype)

        # we convert the mask to a numpy value
        numpy_binary = cls.binary_to_numpy(binary=binary, dtype=dtype)

        # we create a new tensor using the given shape and the fill_value
        array = numpy.full(shape, fill_value=fill_value)
        # we set the indices to the actual numpy_binary value
        # this works as setting a slice of a tensor with a single value
        # broadcasts it onto the whole slice
        array[index] = numpy_binary

        return array

    @classmethod
    def get_numpy_bitwidth(cls, element: numpy.ndarray) -> int:
        return int(cls.NUMPY_DATA_WIDTH_MAPPING[cls.get_numpy_dtype(element)])

    @classmethod
    def get_numpy_dtype(cls, element: numpy.ndarray) -> numpy.dtype:
        return element.dtype

    @classmethod
    def expand_bit_to_numpy_dtype(
            cls,
            bit: bool,
            dtype: numpy.dtype,
            # device is required to match the same interface as cupy
            device: typing.Any = None,
    ) -> numpy.ndarray:
        # we transform the bit into an integer, after checking its boolean
        # value
        bit = int(bool(bit))

        # we get the bitwidth
        bitwidth = int(cls.NUMPY_DATA_WIDTH_MAPPING[dtype])
        # we need the unsigned dtype over integers
        binary_dtype = cls.NUMPY_DATA_CONVERSION_MAPPING[dtype]

        # we create the binary array with a different dtype, from the
        # conversion list, so that we can set the bits properly using
        # Python integers
        # here the array is initialized to all 1s or 0s
        binary_array = numpy.array(bit * 2 ** bitwidth - 1, dtype=binary_dtype)

        # we view it with the final dtype before returning it
        return binary_array.view(dtype)

    # we need this class to convert a numpy array to an array which can be
    # OR/AND/XOR in a bitwise manner, which is a uint representation
    @classmethod
    def numpy_dtype_to_bitwise_numpy(cls, element: numpy.ndarray):
        dtype = cls.get_numpy_dtype(element)
        return element.view(cls.NUMPY_DATA_CONVERSION_MAPPING[dtype])

    # we need this class to convert a numpy array to an array which can be
    # OR/AND/XOR in a bitwise manner, which is a uint representation
    @classmethod
    def bitwise_numpy_to_numpy_dtype(
            cls,
            element: numpy.ndarray,
            dtype: numpy.dtype,
    ):
        return element.view(dtype)
