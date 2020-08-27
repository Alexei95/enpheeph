import typing

import numpy
import torch

from . import operationabc

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


class BitFlipOperation(operationabc.OperationABC):
    @staticmethod
    # gets the binary value from a PyTorch element
    # NOTE: use typing.Mapping for annotating dict as arguments
    def pytorch_element_to_binary(value: torch.Tensor,
                                  *,
                                  data_conversion_mapping: typing.Mapping[numpy.dtype, type] = DATA_CONVERSION_MAPPING,
                                  data_width_mapping: typing.Mapping[numpy.dtype, str] = DATA_WIDTH_MAPPING,
                                  template_string: str = TEMPLATE_STRING):
        # required because shapes (1, ) and () are considered different
        # and we need ()
        if value.size() != tuple():
            value = value[0]

        # we get the numpy value, keeping the same datatype
        numpy_value = value.cpu().numpy()
        dtype = numpy_value.dtype
        # we convert data type
        new_dtype = data_conversion_mapping[dtype]
        # we need the witdth of the new data type
        width = data_width_mapping[dtype]
        # we view the number with a different datatype (int) so we can extract the bits
        str_bin_value = template_string.format(width).format(numpy_value.view(new_dtype))

        return str_bin_value

    @staticmethod
    # original_value is used only for device and datatype conversion
    def binary_to_pytorch_element(binary: str, original_value: torch.Tensor,
                                  *,
                                  data_conversion_mapping: typing.Mapping[numpy.dtype, type] = DATA_CONVERSION_MAPPING,
                                  ):
        # required because shapes (1, ) and () are considered different and we need ()
        if original_value.size() != tuple():
            original_value = original_value[0]

        dtype = original_value.cpu().numpy().dtype
        # we need the converted data type
        new_dtype = data_conversion_mapping[dtype]

        # we convert the bits to numpy integer through Python int for base 2 conversion
        # then we view it back in the original type and convert it to PyTorch
        # square brackets are for creating a numpy.ndarray for PyTorch
        python_int = int(binary, base=2)
        new_numpy_value = new_dtype([python_int]).view(dtype)
        # we use [0] to return a single element
        return torch.from_numpy(new_numpy_value).to(original_value)[0]

    @staticmethod
    def bit_flip(value: torch.Tensor, bit_flip_index: int,
                 *,
                 pytorch_element_to_binary: typing.Callable = 'BitFlipOperation.pytorch_element_to_binary',
                 binary_to_pytorch_element: typing.Callable = 'BitFlipOperation.binary_to_pytorch_element'):
        list_str_bin_value = list(pytorch_element_to_binary(value))

        if list_str_bin_value[bit_flip_index] == '0':
            list_str_bin_value[bit_flip_index] = '1'
        elif list_str_bin_value[bit_flip_index] == '1':
            list_str_bin_value[bit_flip_index] = '0'

        return binary_to_pytorch_element(''.join(list_str_bin_value), value)

    def __call__(self, index, tensor, *args, **kwargs):
        pass
