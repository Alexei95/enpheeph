import numpy
import torch

# gets the binary value from a PyTorch element
def pytorch_element_to_binary(value: torch.Tensor):
    # required because shapes (1, ) and () are considered different and we need ()
    if value.size() != tuple():
        value = value[0]
    # uint to avoid double sign repetition
    data_mapping = {numpy.dtype('float16'): numpy.uint16,
                    numpy.dtype('float32'): numpy.uint32,
                    numpy.dtype('float64'): numpy.uint64}
    width_mapping = {numpy.dtype('float16'): '16',
                     numpy.dtype('float32'): '32',
                     numpy.dtype('float64'): '64'}
    # we get the numpy value, keeping the same datatype
    numpy_value = value.cpu().numpy()
    dtype = numpy_value.dtype
    # we view the number with a different datatype (int) so we can extract the bits
    str_bin_value = '{{:0{}b}}'.format(width_mapping[dtype]).format(numpy_value.view(data_mapping[dtype]))

    return str_bin_value

# original_value is used only for device and datatype conversion
def binary_to_pytorch_element(binary: str, original_value: torch.Tensor):
    # required because shapes (1, ) and () are considered different and we need ()
    if original_value.size() != tuple():
        original_value = original_value[0]

    # uint to avoid double sign repetition
    data_mapping = {numpy.dtype('float16'): numpy.uint16,
                    numpy.dtype('float32'): numpy.uint32,
                    numpy.dtype('float64'): numpy.uint64}
    width_mapping = {numpy.dtype('float16'): '16',
                     numpy.dtype('float32'): '32',
                     numpy.dtype('float64'): '64'}

    dtype = original_value.cpu().numpy().dtype

    # we convert the bits to numpy integer through Python int for base 2 conversion
    # then we view it back in the original type and convert it to PyTorch
    # square brackets are for creating a numpy.ndarray for PyTorch
    new_numpy_value = data_mapping[dtype]([int(binary, base=2)]).view(dtype)
    # we use [0] to return a single element
    return torch.from_numpy(new_numpy_value).to(original_value)[0]

def bit_flip(value: torch.Tensor, n_bit_flips: int):
    list_str_bin_value = list(pytorch_element_to_binary(value))

    perm = torch.randperm(len(list_str_bin_value))

    for i in range(n_bit_flips):
        # also using ^ 1 to invert
        if list_str_bin_value[perm[i]] == '0':
            list_str_bin_value[perm[i]] = '1'
        elif list_str_bin_value[perm[i]] == '1':
            list_str_bin_value[perm[i]] = '0'

    return binary_to_pytorch_element(''.join(list_str_bin_value), value)
