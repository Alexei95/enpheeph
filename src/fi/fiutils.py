import copy

import numpy
import torch

from . import basefaultdescriptor

# uint to avoid double sign repetition
DATA_CONVERSION_MAPPING = {numpy.dtype('float16'): numpy.uint16,
                           numpy.dtype('float32'): numpy.uint32,
                           numpy.dtype('float64'): numpy.uint64,
                           numpy.dtype('uint8'): numpy.uint8,
                           numpy.dtype('int8'): numpy.uint8,
                           numpy.dtype('int16'): numpy.uint16,
                           numpy.dtype('int32'): numpy.uint32,
                           numpy.dtype('int64'): numpy.uint64,
                           }
DATA_WIDTH_MAPPING = {numpy.dtype('float16'): '16',
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
TEMPLATE_STRING = '{{:0{}b}}'


# gets the binary value from a PyTorch element
def pytorch_element_to_binary(value: torch.Tensor) -> str:
    # required because shapes (1, ) and () are considered different and we need ()
    if value.size() != tuple():
        value = value[0]

    # we get the numpy value, keeping the same datatype
    numpy_value = value.cpu().numpy()
    dtype = numpy_value.dtype
    # we convert data type
    new_dtype = DATA_CONVERSION_MAPPING[dtype]
    # we need the witdth of the new data type
    width = DATA_WIDTH_MAPPING[dtype]
    # we view the number with a different datatype (int) so we can extract the bits
    str_bin_value = TEMPLATE_STRING.format(width).format(numpy_value.view(new_dtype))

    return str_bin_value


def inject_fault_binary(binary: str,
                        fault: basefaultdescriptor.BaseFaultDescriptor,
                        sampler: torch.Generator = None) -> str:
    injected_binary = copy.deepcopy(binary)
    for index in fault.bit_index:
        if fault.bit_value == basefaultdescriptor.BitValue.One:
            injected_binary[index] = "1"
        elif fault.bit_value == basefaultdescriptor.BitValue.Zero:
            injected_binary[index] = "0"
        elif fault.bit_value == basefaultdescriptor.BitValue.BitFlip:
            injected_binary[index] = str(int(injected_binary[index]) ^ 1)
        elif fault.bit_value == basefaultdescriptor.BitValue.Random:
            # if we do not have a sampler
            if sampler is None:
                raise ValueError("A sampler must be passed when using random bit-flips")
                # SAMPLER_SEED = 2147483647
                # sampler = torch.Generator(device='cpu')
                # sampler.manual_seed(SAMPLER_SEED)
            random_bit = torch.randint(0, 2, size=(), generator=sampler)
            injected_binary[index] = str(random_bit.item())
    return injected_binary


# original_value is used only for device and datatype conversion
def binary_to_pytorch_element(binary: str, original_value: torch.Tensor) -> torch.Tensor:
    # required because shapes (1, ) and () are considered different and we need ()
    if original_value.size() != tuple():
        original_value = original_value[0]

    dtype = original_value.cpu().numpy().dtype
    # we need the converted data type
    new_dtype = DATA_CONVERSION_MAPPING[dtype]

    # we convert the bits to numpy integer through Python int for base 2 conversion
    # then we view it back in the original type and convert it to PyTorch
    # square brackets are for creating a numpy.ndarray for PyTorch
    python_int = int(binary, base=2)
    new_numpy_value = new_dtype([python_int]).view(dtype)
    # we use [0] to return a single element
    return torch.from_numpy(new_numpy_value).to(original_value)[0]


def inject_fault_pytorch(tensor: torch.Tensor,
                         fault: basefaultdescriptor.BaseFaultDescriptor,
                         sampler: torch.Generator = None) -> torch.Tensor:
    binary = pytorch_to_binary(tensor)
    injected_binary = inject_fault_binary(binary, fault, sampler)
    injected_tensor = binary_to_pytorch_element(binary, tensor)
    return injected_tensor
