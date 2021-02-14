import dataclasses
import enum
import typing


ParameterType = enum.Enum('ParameterType', 'Weight Activation', module=__name__)
BitValue = enum.Enum('BitValue', 'Random Zero One BitFlip', module=__name__)


# this is a container for the module name and the index for where the fault
# should be injected
# each fault descriptor covers a single bit-flip (or stuck-at)
@dataclasses.dataclass(init=True)
class BaseFaultDescriptor(object):
    # name of the module to inject
    module_name: str
    # name of the parameter, if the module is a base module (conv, fc),
    # it generally coincides with 'weight' for weight injection
    # not required for activation injection
    parameter_name: str = None
    # type of parameter to inject, weight, activation, ...
    parameter_type: ParameterType
    # index of the tensor to be injected
    tensor_index: tuple[typing.Union[int, slice]]
    # index of the bit to be injected
    # even if it is a slice, it must be converted to indices
    # just use tuple(range(*slice.indices(slice.stop)))
    bit_index: tuple[int]
    # type of bit injection to be carried out
    bit_value: BitValue

    def __post_init__(self):
        # FIXME: here there should be some checks on the values
        pass

    @staticmethod
    def bit_index_from_slice(self, slice_: slice):
        return tuple(range(*slice.indices(slice.stop)))
