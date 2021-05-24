import copy

import numpy
import torch

import src.fi.injection.faultdescriptor

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
    # required because shapes (1, ) and () are considered different and we need
    # ()
    if value.size() != tuple():
        value = value[0]

    # we get the numpy value, keeping the same datatype
    numpy_value = value.cpu().numpy()
    dtype = numpy_value.dtype
    # we convert data type
    new_dtype = DATA_CONVERSION_MAPPING[dtype]
    # we need the witdth of the new data type
    width = DATA_WIDTH_MAPPING[dtype]
    # we view the number with a different datatype (int) so we can extract the
    # bits
    str_bin_value = TEMPLATE_STRING.format(
            width
    ).format(
            numpy_value.view(
                    new_dtype
            )
    )

    return str_bin_value


def inject_fault_binary(binary: str,
                        fault: src.fi.injection.faultdescriptor.FaultDescriptor,
                        sampler: torch.Generator = None) -> str:
    # we need to convert the binary string into a list of characters
    # otherwise we cannot update the values
    injected_binary = list(copy.deepcopy(binary))
    for index in fault.bit_index_conversion(
            bit_index=fault.bit_index,
            bit_width=len(injected_binary),
            ):
        # if we are using little endian we invert the index, as the LSB is
        # at the end of the list
        if fault.endianness == fault.endianness.Little:
            index = (len(injected_binary) - 1) - index

        if fault.bit_value == fault.bit_value.StuckAtOne:
            injected_binary[index] = "1"
        elif fault.bit_value == fault.bit_value.StuckAtZero:
            injected_binary[index] = "0"
        elif fault.bit_value == fault.bit_value.BitFlip:
            injected_binary[index] = str(int(injected_binary[index]) ^ 1)
        elif fault.bit_value == fault.bit_value.Random:
            # if we do not have a sampler
            if sampler is None:
                raise ValueError(
                        "A sampler must be passed "
                        "when using random bit-flips"
                )
                # SAMPLER_SEED = 2147483647
                # sampler = torch.Generator(device='cpu')
                # sampler.manual_seed(SAMPLER_SEED)
            random_bit = torch.randint(0, 2, size=(), generator=sampler)
            injected_binary[index] = str(random_bit.item())
    return ''.join(injected_binary)


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
                         fault: src.fi.injection.faultdescriptor.FaultDescriptor,
                         sampler: torch.Generator = None) -> torch.Tensor:
    binary = pytorch_element_to_binary(tensor)
    injected_binary = inject_fault_binary(binary, fault, sampler)
    injected_tensor = binary_to_pytorch_element(injected_binary, tensor)
    return injected_tensor


def inject_tensor_fault_pytorch(
        tensor: torch.Tensor,
        fault: src.fi.injection.faultdescriptor.FaultDescriptor,
        sampler: torch.Generator = None) -> torch.Tensor:
    # we deepcopy the tensor to avoid modifying the original one
    tensor = copy.deepcopy(tensor)
    # we get the tensor value to be injected
    # first we convert the fault tensor index to a proper tensor index for the
    # tensor case
    original_tensor = tensor[
            fault.tensor_index_conversion(
                    tensor_index=fault.tensor_index,
                    tensor_shape=tensor.size()
            )
    ]
    # to inject the values, we need to flatten the tensor
    flattened_tensor = original_tensor.flatten()
    # then we need to process them one by one, by injecting the faults
    # the returned elements are tensors
    injected_flattened_tensor_list = []
    for element in flattened_tensor:
        injected_flattened_tensor_list.append(
                src.fi.fiutils.inject_fault_pytorch(element, fault))
    # # we create a list with the injected data, converting back to tensors
    # injected_flattened_tensor_list = []
    # for injected_binary, original_binary in zip(
    #         injected_flattened_bit_tensor, flattened_tensor):
    #     injected_flattened_tensor_list.append(
    #             src.fi.fiutils.binary_to_pytorch_element(
    #                     injected_binary, original_binary))
    # print(injected_flattened_tensor_list)
    # we create a tensor from the list, moving it to the same device as the
    # original one
    injected_flattened_tensor = torch.Tensor(
            injected_flattened_tensor_list).to(flattened_tensor)
    # we reshape the tensor to the original one
    injected_tensor = injected_flattened_tensor.reshape(original_tensor.size())
    # we update the tensor to the new value
    tensor[fault.tensor_index_conversion(
            fault.tensor_index,
            tensor.size()
    )] = injected_tensor

    # add copy.deepcopy for more redundancy, to avoid modifying the original
    # one
    return copy.deepcopy(tensor)
