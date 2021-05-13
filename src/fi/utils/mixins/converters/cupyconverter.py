import copy
import functools
import operator
import typing

import cupy


class CupyConverter(object):
    # uint to avoid double sign repetition
    CUPY_DATA_CONVERSION_MAPPING = {
            cupy.dtype('float16'): cupy.uint16,
            cupy.dtype('float32'): cupy.uint32,
            cupy.dtype('float64'): cupy.uint64,
            cupy.dtype('uint8'): cupy.uint8,
            cupy.dtype('int8'): cupy.uint8,
            cupy.dtype('int16'): cupy.uint16,
            cupy.dtype('int32'): cupy.uint32,
            cupy.dtype('int64'): cupy.uint64,
    }
    # FIXME: automate this creation using nbytes of a basic tensor like [0]
    # it could incur into issues
    CUPY_DATA_WIDTH_MAPPING = {
            cupy.dtype('float16'): '16',
            cupy.dtype('float32'): '32',
            cupy.dtype('float64'): '64',
            cupy.dtype('uint8'): '8',
            cupy.dtype('int8'): '8',
            cupy.dtype('int16'): '16',
            cupy.dtype('int32'): '32',
            cupy.dtype('int64'): '64',
    }
    # this template first requires the width (the single {}) and then it can
    # convert a number to a binary view using that width and filling the extra
    # on the left with 0s
    CUPY_TEMPLATE_STRING = '{{:0{}b}}'

    @classmethod
    def single_cupy_to_binary(cls, element: cupy.ndarray) -> str:
        # if we have different than 1 element, we raise ValueError
        if element.size != 1:
            # FIXME: check which error to raise
            raise ValueError('There must be only 1 element in the array')

        # we remove the extra dimensions using squeeze
        element = element.squeeze()
        # we get the dtype of the element
        dtype = element.dtype
        # we convert data type
        new_dtype = cls.CUPY_DATA_CONVERSION_MAPPING[dtype]
        # we need the width of the new data type
        width = cls.CUPY_DATA_WIDTH_MAPPING[dtype]
        # we view the number with a different datatype (int) so we can extract
        # the bits
        str_bin_value = cls.CUPY_TEMPLATE_STRING.format(
                width
        ).format(
                element.view(
                        new_dtype
                )
        )

        return str_bin_value

    # the outcome will be flattened, to be compatible with Python lists
    @classmethod
    def cupy_to_binary(cls, element: cupy.ndarray) -> typing.List[str]:
        binaries = []
        for el in element.flatten():
            binaries.append(cls.single_cupy_to_binary(el))
        return binaries

    # we need the cupy dtype to return the data in
    # the returned array has () shape, it is a single element
    @classmethod
    def binary_to_cupy(
            cls,
            binary: str,
            dtype: cupy.dtype,
            device: cupy.cuda.Device = None,
    ) -> cupy.ndarray:
        # we get the default device if it is None
        if device is None:
            # by default it is Device 0
            device = cupy.cuda.Device()

        # we need the converted data type, to convert it back to the original
        conv_dtype = cls.CUPY_DATA_CONVERSION_MAPPING[dtype]

        # we convert the bits to cupy integer through Python int for base 2
        # conversion
        # then we view it back in the original type and convert it to PyTorch
        # square brackets are for creating a cupy.ndarray for PyTorch
        python_int = int(binary, base=2)
        with device:
            new_cupy_value = conv_dtype(python_int).view(dtype)

        return new_cupy_value

    # this method replicates the same binary value across a set of indices,
    # given also the overall tensor shape
    # fill_value is used to fill the non-selected values in the tensor
    @classmethod
    def binary_to_cupy_broadcast(
            cls,
            binary: str,
            dtype: cupy.dtype,
            index: typing.Sequence[typing.Union[int, slice]],
            shape: typing.Sequence[int],
            fill_value: cupy.ndarray = cupy.array(0),
            device: cupy.cuda.Device = None,
    ):
        # we convert the fill_value to a numpy array with the correct dtype
        fill_value = cupy.array(fill_value, dtype=dtype)

        # we convert the mask to a numpy value
        cupy_binary = cls.binary_to_cupy(
                binary=binary,
                dtype=dtype,
                device=device
        )

        # we create a new tensor using the given shape and the fill_value
        array = cupy.full(shape, fill_value=fill_value)
        # we set the indices to the actual cupy_binary value
        # this works as setting a slice of a tensor with a single value
        # broadcasts it onto the whole slice
        array[index] = cupy_binary

        return array

    @classmethod
    def get_cupy_bitwidth(cls, element: cupy.ndarray) -> int:
        return int(cls.CUPY_DATA_WIDTH_MAPPING[cls.get_cupy_dtype(element)])

    @classmethod
    def get_cupy_dtype(cls, element: cupy.ndarray) -> cupy.dtype:
        return element.dtype

    @classmethod
    def get_cupy_device(cls, element: cupy.ndarray) -> cupy.cuda.Device:
        return element.device

    @classmethod
    def expand_bit_to_cupy_dtype(
            cls,
            bit: bool,
            dtype: cupy.dtype,
            device: cupy.cuda.Device = None,
    ) -> cupy.ndarray:
        # we transform the bit into an integer, after checking its boolean
        # value
        bit = int(bool(bit))
        # we get the default device if it is None
        if device is None:
            # by default it is Device 0
            device = cupy.cuda.Device()

        # we get the bitwidth
        bitwidth = int(cls.CUPY_DATA_WIDTH_MAPPING[dtype])
        # we need the unsigned dtype over integers
        binary_dtype = cls.CUPY_DATA_CONVERSION_MAPPING[dtype]

        # we create the binary array with a different dtype, from the
        # conversion list, so that we can set the bits properly using
        # Python integers
        # here the array is initialized to all 1s or 0s
        with device:
            binary_array = cupy.array(
                    bit * 2 ** bitwidth - 1,
                    dtype=binary_dtype,
            )

        # we view it with the final dtype before returning it
        return binary_array.view(dtype)

    # we need this class to convert a cupy array to an array which can be
    # OR/AND/XOR in a bitwise manner, which is a uint representation
    @classmethod
    def cupy_dtype_to_bitwise_cupy(cls, element: cupy.ndarray):
        dtype = cls.get_cupy_dtype(element)
        return element.view(cls.CUPY_DATA_CONVERSION_MAPPING[dtype])

    # we need this class to convert a cupy array to an array which can be
    # OR/AND/XOR in a bitwise manner, which is a uint representation
    @classmethod
    def bitwise_cupy_to_cupy_dtype(
            cls,
            element: cupy.ndarray,
            dtype: cupy.dtype,
    ):
        return element.view(dtype)

    # we can use this method for checking the availability of CUDA
    @classmethod
    def cuda_support(cls):
        return cupy is not None and cupy.cuda.is_available()
