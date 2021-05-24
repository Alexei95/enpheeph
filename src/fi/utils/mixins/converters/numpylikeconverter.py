import copy
import functools
import operator
import typing

import src.utils.mixins.classutils
import src.utils.mixins.dispatcher
import src.fi.utils.mixins.converters.numpyconverter
import src.fi.utils.mixins.converters.cupyconverter


class NumpyLikeConverter(
        src.utils.mixins.classutils.ClassUtils,
        src.utils.mixins.dispatcher.Dispatcher,
        src.fi.utils.mixins.converters.numpyconverter.NumpyConverter,
        src.fi.utils.mixins.converters.cupyconverter.CupyConverter,
):
    NUMPY_STRING = 'numpy'
    CUPY_STRING = (
            'cupy'
            if src.fi.utils.mixins.converters.cupyconverter.
                    CupyConverter.cuda_support()
            else None
    )

    # FIXME: check whether these methods can be moved
    # this method is used to remove all the associations after calling a
    # classmethod, made especially for the libraries to Pytorch associations
    @classmethod
    def _deregister_libraries(cls):
        cls.deregister(cls.NUMPY_STRING)
        cls.deregister(cls.CUPY_STRING)

    @classmethod
    def single_numpy_like_to_binary(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray']
    ) -> str:
        cls.register(cls.NUMPY_STRING, cls.single_numpy_to_binary)
        cls.register(cls.CUPY_STRING, cls.single_cupy_to_binary)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
        )

        cls._deregister_libraries()

        return out

    # the outcome will be flattened, to be compatible with Python lists
    @classmethod
    def numpy_like_to_binary(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
    ) -> typing.List[str]:
        cls.register(cls.NUMPY_STRING, cls.numpy_to_binary)
        cls.register(cls.CUPY_STRING, cls.cupy_to_binary)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
        )

        cls._deregister_libraries()

        return out

    # we need the cupy dtype to return the data in
    # the returned array has () shape, it is a single element
    @classmethod
    def binary_to_numpy_like(
            cls,
            binary: str,
            dtype: typing.Union['numpy.dtype', 'cupy.dtype'],
            library: str,
            device: 'cupy.cuda.Device' = None,
    ) -> typing.Union['numpy.ndarray', 'cupy.ndarray']:
        cls.register(cls.NUMPY_STRING, cls.binary_to_numpy)
        cls.register(cls.CUPY_STRING, cls.binary_to_cupy)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                library,
                binary=binary,
                dtype=dtype,
                device=device,
        )

        cls._deregister_libraries()

        return out

    # this method replicates the same binary value across a set of indices,
    # given also the overall tensor shape
    # fill_value is used to fill the non-selected values in the tensor
    @classmethod
    def binary_to_numpy_like_broadcast(
            cls,
            binary: str,
            dtype: typing.Union['numpy.dtype', 'cupy.dtype'],
            index: typing.Sequence[typing.Union[int, slice]],
            shape: typing.Sequence[int],
            library: str,
            # here we use an integer but a custom ndarray may preserve more
            # information
            fill_value: typing.Union['numpy.ndarray', 'cupy.ndarray', int] = 0,
            device: 'cupy.cuda.Device' = None,
    ):
        cls.register(cls.NUMPY_STRING, cls.binary_to_numpy_broadcast)
        cls.register(cls.CUPY_STRING, cls.binary_to_cupy_broadcast)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                library,
                binary=binary,
                dtype=dtype,
                index=index,
                shape=shape,
                fill_value=fill_value,
                device=device,
        )

        cls._deregister_libraries()

        return out

    @classmethod
    def get_numpy_like_bitwidth(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
    ) -> int:
        cls.register(cls.NUMPY_STRING, cls.get_numpy_bitwidth)
        cls.register(cls.CUPY_STRING, cls.get_cupy_bitwidth)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
        )

        cls._deregister_libraries()

        return out

    @classmethod
    def get_numpy_like_dtype(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
    ) -> typing.Union['cupy.dtype', 'numpy.dtype']:
        cls.register(cls.NUMPY_STRING, cls.get_numpy_dtype)
        cls.register(cls.CUPY_STRING, cls.get_cupy_dtype)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
        )

        cls._deregister_libraries()

        return out

    @classmethod
    def get_numpy_like_device(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
    ) -> typing.Union['cupy.cuda.Device', str]:
        if cls.get_main_library_from_object(element) == cls.CUPY_STRING:
            return cls.get_cupy_device(element)
        else:
            return 'cpu'

    @classmethod
    def expand_bit_to_numpy_like_dtype(
            cls,
            bit: bool,
            dtype: typing.Union['numpy.dtype', 'cupy.dtype'],
            library: str,
            device: 'cupy.cuda.Device' = None,
    ) -> typing.Union['numpy.ndarray', 'cupy.ndarray']:
        cls.register(cls.NUMPY_STRING, cls.expand_bit_to_numpy_dtype)
        cls.register(cls.CUPY_STRING, cls.expand_bit_to_cupy_dtype)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                library,
                bit=bit,
                dtype=dtype,
                device=device,
        )

        cls._deregister_libraries()

        return out

    # we need this class to convert an array to an array which can be
    # OR/AND/XOR in a bitwise manner, which is a uint representation
    @classmethod
    def numpy_like_dtype_to_bitwise_numpy_like(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
    ):
        cls.register(cls.NUMPY_STRING, cls.numpy_dtype_to_bitwise_numpy)
        cls.register(cls.CUPY_STRING, cls.cupy_dtype_to_bitwise_cupy)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
        )

        cls._deregister_libraries()

        return out

    # we need this class to convert an array to an array which can be
    # OR/AND/XOR in a bitwise manner, which is a uint representation
    @classmethod
    def bitwise_numpy_like_to_numpy_like_dtype(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            dtype: typing.Union['numpy.dtype', 'cupy.dtype'],
    ):
        cls.register(cls.NUMPY_STRING, cls.bitwise_numpy_to_numpy_dtype)
        cls.register(cls.CUPY_STRING, cls.bitwise_cupy_to_cupy_dtype)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
                dtype=dtype,
        )

        cls._deregister_libraries()

        return out

    # this method takes an object from numpy/cupy and returns the corresponding
    # string, which is defined from NUMPY_STRING or CUPY_STRING
    # NOTE: this method could be substituted with get_main_library_from_object,
    # but in this case it is more generic, at least for now
    @classmethod
    def get_numpy_like_string(cls, object_: typing.Any) -> str:
        cls.register(cls.NUMPY_STRING, lambda x: cls.NUMPY_STRING)
        cls.register(cls.CUPY_STRING, lambda x: cls.CUPY_STRING)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(object_),
                object_,
        )

        cls._deregister_libraries()

        return out

    # this method gets the bitwidth associated to the dtype
    @classmethod
    def numpy_like_bitwidth_from_dtype(
            cls,
            dtype: typing.Union['numpy.dtype', 'cupy.dtype'],
            library: str
    ) -> int:
        cls.register(cls.NUMPY_STRING, cls.get_numpy_bitwidth_from_dtype)
        cls.register(cls.CUPY_STRING, cls.get_cupy_bitwidth_from_dtype)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                library,
                dtype=dtype,
        )

        cls._deregister_libraries()

        return out

    # this classmethod tries to convert a generic element to an array of a
    # specified type
    @classmethod
    def to_numpy_like_array(
            cls,
            element: typing.Any,
            dtype: typing.Union['numpy.dtype', 'cupy.dtype'],
            library: str,
            device: 'cupy.cuda.Device' = None,
    ) -> typing.Union['numpy.ndarray', 'cupy.ndarray']:
        cls.register(cls.NUMPY_STRING, cls.to_numpy_array)
        cls.register(cls.CUPY_STRING, cls.to_cupy_array)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                library,
                element=element,
                dtype=dtype,
                device=device,
        )

        cls._deregister_libraries()

        return out

    # here we can broadcast a value in single-shaped array to a full shape
    # covering the non-indexed values with a determined fill_value
    @classmethod
    def numpy_like_broadcast(
            cls,
            element: typing.Union['numpy.ndarray', 'cupy.ndarray'],
            index: typing.Sequence[typing.Union[int, slice]],
            shape: typing.Sequence[int],
            fill_value: typing.Union['numpy.ndarray', 'cupy.ndarray', int] = 0,
    ) -> typing.Union['numpy.ndarray', 'cupy.ndarray']:
        cls.register(cls.NUMPY_STRING, cls.numpy_broadcast)
        cls.register(cls.CUPY_STRING, cls.cupy_broadcast)

        # we dispatch the conversion to the correct handler
        out = cls.dispatch_call(
                cls.get_main_library_from_object(element),
                element=element,
                index=index,
                shape=shape,
                fill_value=fill_value,
        )

        cls._deregister_libraries()

        return out