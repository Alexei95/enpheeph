import numpy
import torch

from ..samplers.basesampler import BaseSampler

# uint to avoid double sign repetition
DATA_CONVERSION_MAPPING = {numpy.dtype('float16'): numpy.uint16,
                numpy.dtype('float32'): numpy.uint32,
                numpy.dtype('float64'): numpy.uint64}
DATA_WIDTH_MAPPING = {numpy.dtype('float16'): '16',
                    numpy.dtype('float32'): '32',
                    numpy.dtype('float64'): '64'}
# this template first requires the width (the single {}) and then it can
# convert a number to a binary view using that width and filling the extra
# on the left with 0s
TEMPLATE_STRING = '{{:0{}b}}'


# gets the binary value from a PyTorch element
def pytorch_element_to_binary(value: torch.Tensor):
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


# original_value is used only for device and datatype conversion
def binary_to_pytorch_element(binary: str, original_value: torch.Tensor):
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


def bit_flip(value: torch.Tensor, n_bit_flips: int, sampler: BaseSampler):
    list_str_bin_value = list(pytorch_element_to_binary(value))

    perm = torch.randperm(len(list_str_bin_value))

    for i in sampler.iter_choice(low=0, high=len(list_str_bin_value), ):
        # also using ^ 1 to invert, but it requires int
        if list_str_bin_value[perm[i]] == '0':
            list_str_bin_value[perm[i]] = '1'
        elif list_str_bin_value[perm[i]] == '1':
            list_str_bin_value[perm[i]] = '0'

    return binary_to_pytorch_element(''.join(list_str_bin_value), value)
