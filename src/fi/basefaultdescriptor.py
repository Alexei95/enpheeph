import collections.abc
import dataclasses
import enum
import typing


class Endianness(enum.Flag):
    Little = enum.auto()
    Big = enum.auto()

    MSBAtIndexZero = Big
    LSBAtIndexZero = Little


class ParameterType(enum.Flag):
    Weight = enum.auto()
    Activation = enum.auto()
    Sparse = enum.auto()
    COO = enum.auto()
    Index = enum.auto()
    Value = enum.auto()
    SparseWeightCOOIndex = Sparse | Weight | COO | Index
    SparseWeightCOOValue = Sparse | Weight | COO | Value
    SparseActivationCOOIndex = Sparse | Activation | COO | Index
    SparseActivationCOOValue = Sparse | Activation | COO | Value


class BitValue(enum.Enum):
    Random = enum.auto()
    StuckAtZero = enum.auto()
    StuckAtOne = enum.auto()
    BitFlip = enum.auto()


# this is a container for the module name and the index for where the fault
# should be injected
# each fault descriptor covers a single bit-flip (or stuck-at)
# we need unsafe hash for using it as a dictonary key
@dataclasses.dataclass(init=True, repr=True, unsafe_hash=True)
class BaseFaultDescriptor(object):
    # name of the module to inject
    module_name: str
    # type of parameter to inject, weight, activation, ...
    parameter_type: ParameterType
    # index of the tensor to be injected, it will be converted to a tuple
    tensor_index: typing.Union[
            type(Ellipsis),
            typing.Sequence[typing.Union[int, slice, type(Ellipsis)]]
    ]
    # index of the bit to be injected
    # even if it is a slice, it must be converted to indices
    # just use tuple(range(*slice.indices(slice.stop)))
    bit_index: typing.Union[typing.Sequence[int], slice]
    # type of bit injection to be carried out
    bit_value: BitValue
    # way to interpret the binary representation, as big endian or little
    # endian
    # big endian means the bit index 0 is mapped to the MSB of the binary
    # little endian means the bit index 0 is instead mapped to the LSB
    # by default we use little endian, as this is the most common
    # representation
    endianness: Endianness = Endianness.Little
    # name of the parameter, if the module is a base module (conv, fc),
    # it generally coincides with 'weight' for weight injection
    # not required for activation injection
    parameter_name: str = None

    def __post_init__(self):
        # FIXME: here there should be some checks on the values

        # we raise error if parameter_name is not set and we are not doing
        # activation injection
        activation_flag = self.parameter_type.Activation in self.parameter_type
        if self.parameter_name is None and not activation_flag:
            raise ValueError('Please provide a parameter_name from which '
                    'to gather the values of the tensor to be injected')

        # we convert the tensor index to a tuple, otherwise the descriptor
        # remains unhashable
        # first we have to check if it is a MutableSequence
        if isinstance(self.tensor_index, collections.abc.MutableSequence):
            self.tensor_index = tuple(self.tensor_index)

        # if the bit index is a slice, we convert it to indices
        if isinstance(self.bit_index, slice):
            self.bit_index = self.bit_index_from_slice(self.bit_index)

    @staticmethod
    def bit_index_from_slice(slice_: slice) -> typing.Tuple[int, ...]:
        # we use slice.indices to get back the indices of the slice
        # we must pass a stop as it requires a stop value, taken from the slice
        # itself
        # in this way we can return all of the affected indices in a tuple
        return tuple(range(*slice_.indices(slice_.stop)))

    @staticmethod
    def to_tensor_slice(
                tensor_index: typing.Union[
                        type(Ellipsis),
                        typing.Sequence[
                                typing.Union[
                                        int, slice, type(Ellipsis)]]],
                tensor_shape: typing.Sequence[int],
            ) -> typing.Sequence[typing.Union[int, slice]]:
        # if we have a single ellipsis, without any container, we have a slice
        # covering the whole tensor shape
        if isinstance(tensor_index, type(Ellipsis)):
            return tuple(slice(0, dim_range) for dim_range in tensor_shape)
        # we check if the number of elements in the index and in the number of
        # dimensions is the same, otherwise we raise ValueError
        # this check exists as zip goes over the shortest one
        if len(tensor_index) != len(tensor_shape):
            raise ValueError(
                    "Number of elements in the index must be the "
                    "same as the number of dimensions in the tensor"
            )
        new_tensor_index = []
        for index, dim_range in zip(tensor_index, tensor_shape):
            # if the current index is a slice
            # then we limit the ending of the slice to the maximum range
            # of the current dimension
            if isinstance(index, slice):
                new_index = index.indices(dim_range - 1)
            # if we get an Ellipsis, then we set it up as a slice from 0 to
            # max dimension range
            elif isinstance(index, type(Ellipsis)):
                new_index = slice(0, dim_range - 1)
            # if it is int we copy it
            elif isinstance(index, int):
                new_index = index
            # as fallback we raise ValueError
            else:
                raise ValueError('Wrong index value, use slice, int or ...')
            new_tensor_index.append(new_index)
        return tuple(new_tensor_index)
