import collections.abc
import dataclasses
import typing

import src.fi.utils.enums.bitvalue
import src.fi.utils.enums.endianness
import src.fi.utils.enums.parametertype

# FIXME: fix the way hash is computed, as this solution is very sketchy
# a better idea could be to use frozen
# this is a container for the module name and the index for where the fault
# should be injected
# each fault descriptor covers a single bit-flip (or stuck-at)
# we need unsafe hash for using it as a dictonary key
@dataclasses.dataclass(
        init=True, repr=True, unsafe_hash=True, eq=True
)
class FaultDescriptor(object):
    # name of the module to inject
    module_name: str
    # type of parameter to inject, weight, activation, ...
    parameter_type: src.fi.utils.enums.parametertype.ParameterType
    # index of the tensor to be injected, it will be converted to a tuple
    tensor_index: typing.Union[
            type(Ellipsis),
            typing.Sequence[typing.Union[int, slice, type(Ellipsis)]]
    ] = dataclasses.field(
            init=True,
            repr=True,
            hash=False
    )
    # the following is a representation of the tensor index, used for
    # hashing purposes
    _tensor_index_repr: str = dataclasses.field(
            init=False, repr=False, compare=False, hash=True
    )
    # index of the bit to be injected
    # even if it is a slice, it must be converted to indices
    # just use tuple(range(*slice.indices(slice.stop)))
    # we support also the ellipsis meaning all the bits
    bit_index: typing.Union[
            typing.Sequence[int], slice, type(Ellipsis)
    ] = dataclasses.field(
            init=True,
            repr=True,
            hash=False
    )
    # type of bit injection to be carried out
    bit_value: src.fi.utils.enums.bitvalue.BitValue

    # way to interpret the binary representation, as big endian or little
    # endian
    # big endian means the bit index 0 is mapped to the MSB of the binary
    # little endian means the bit index 0 is instead mapped to the LSB
    # by default we use little endian, as this is the most common
    # representation
    endianness: src.fi.utils.enums.endianness.Endianness = dataclasses.field(
            init=True,
            repr=True,
            default=src.fi.utils.enums.endianness.Endianness.Little)
    # name of the parameter, if the module is a base module (conv, fc),
    # it generally coincides with 'weight' for weight injection
    # not required for activation injection
    parameter_name: str = dataclasses.field(init=True, repr=True, default=None)

    def __post_init__(self):
        # FIXME: here there should be some checks on the values

        # we raise error if parameter_name is not set and we are not doing
        # activation injection
        activation_flag = self.parameter_type.Activation in self.parameter_type
        snn_state = self.parameter_type.SNN | self.parameter_type.State
        snn_flag = snn_state in self.parameter_type
        flags = not activation_flag and not snn_flag
        if self.parameter_name is None and flags:
            raise ValueError('Please provide a parameter_name from which '
                    'to gather the values of the tensor to be injected')

        # NOTE: for hashability, slice is unhashable, but we use an internal
        # representation to make them hashable

        # we convert the tensor index to a tuple, otherwise the descriptor
        # remains unhashable
        # first we have to check if it is a MutableSequence
        if isinstance(self.tensor_index, collections.abc.MutableSequence):
            self.tensor_index = tuple(self.tensor_index)
        # we set the internal representation for hashability
        self._tensor_index_repr = repr(self.tensor_index)

        # if the bit_index is a ordered container
        # we need to save it as tuple to allow for hashability
        if isinstance(self.bit_index, collections.abc.MutableSequence):
            self.bit_index = tuple(self.bit_index)
        # we set the internal representation for hashability
        self._bit_index_repr = repr(self.bit_index)

    @staticmethod
    def bit_index_conversion(
                bit_index: typing.Union[
                        typing.Sequence[int], slice, type(Ellipsis)],
                bit_width: int,
                ) -> typing.Tuple[int, ...]:
        # if the bit_slice is an Ellipsis, we return a tuple covering all the
        # bits
        if isinstance(bit_index, type(Ellipsis)):
            return tuple(range(bit_width))
        # else we check whether it is a slice
        # in this case we convert the slice to a tuple
        elif isinstance(bit_index, slice):
            # we use slice.indices to get back the indices of the slice
            # we must pass a stop as it requires a stop value, taken from the
            # bitwidth, as the slice chooses the smallest of the internal
            # stop value and from the argument one
            # in this way we can return all of the affected indices in a tuple
            return tuple(range(*bit_index.indices(bit_width)))
        # if instead it is a sequence
        elif isinstance(bit_index, collections.abc.Sequence):
            # we remove all duplicates by constructing a set
            # we sort the list
            # we filter it to be between 0 (included) and the bit_width
            # (excluded)
            return tuple(
                    i
                    for i in sorted(set(bit_index))
                    if 0 <= i < bit_width
            )

    # NOTE: in our case here, Ellipsis (...) is used as non-greedy, so you need
    # one ... per dimension you want to skip. In numpy instead ... is greedy,
    # so it tries to cover as many dimensions as possible. While this behaviour
    # may be nicer, the user of this function should still know or be able to
    # access the information regarding the tensor shape, hence a greedy
    # behaviour may limit customizability in the fault. Therefore, for now the
    # behaviour will remain non-greedy
    @staticmethod
    def tensor_index_conversion(
                tensor_index: typing.Union[
                        type(Ellipsis),
                        typing.Sequence[
                                typing.Union[
                                        int, slice, type(Ellipsis)]]],
                tensor_shape: typing.Sequence[int],
                # if force_index is True we return a list of indices
                # not a slice
                force_index: typing.Optional[bool] = False,
                ) -> typing.Tuple[typing.Union[int, slice]]:
        # if we have a single ellipsis, without any container, we have a slice
        # covering the whole tensor shape
        if isinstance(tensor_index, type(Ellipsis)):
            slices = (slice(0, dim_range) for dim_range in tensor_shape)
            if force_index:
                return tuple(
                        tuple(range(
                                s.start if s.start else 0,
                                s.stop,
                                s.step if s.step else 1
                        ))
                        for s in slices
                )
            else:
                return tuple(slices)
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
                if force_index:
                    new_index = tuple(range(*new_index))
            # if we get an Ellipsis, then we set it up as a slice from 0 to
            # max dimension range
            elif isinstance(index, type(Ellipsis)):
                new_index = slice(0, dim_range - 1)
                if force_index:
                    new_index = tuple(range(
                            new_index.start if new_index.start else 0,
                            new_index.stop,
                            new_index.step if new_index.step else 1
                    ))
            # if it is int we copy it in a tuple
            elif isinstance(index, int):
                new_index = (index, )
            # as fallback we raise ValueError
            else:
                raise ValueError('Wrong index value, use slice, int or ...')
            new_tensor_index.append(new_index)
        # we always return a tuple
        return tuple(new_tensor_index)
