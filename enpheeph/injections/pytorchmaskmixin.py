import abc
import sys
import typing

try:
    import cupy
except ImportError:
    cupy = None
try:
    import numpy
except ImportError:
    numpy = None
import torch

import enpheeph.injections.pytorchinjectionabc
import enpheeph.utils.data_classes
import enpheeph.utils.typings


class PyTorchMaskMixIn(abc.ABC):
    TORCH_CONVERSION = {
            'cpu': numpy.asarray if numpy is not None else None,
            'cuda': lambda x: cupy.fromDlpack(torch.utils.dlpack.to_dlpack(x)) if cupy is not None else None,
    }
    TORCH_RECONVERSION = {
            'cpu': torch.from_numpy,
            'cuda': torch.as_tensor,
    }
    LIBRARY = {
            'cpu': numpy,
            'cuda': cupy,
    }
    MASK_CREATION = {
            'cpu': numpy.array if numpy is not None else None,
            'cuda': cupy.array if cupy is not None else None,
    }

    # the used variables in the functions, must be initialized properly
    fault_locator: enpheeph.utils.data_classes.FaultLocation
    mask: enpheeph.utils.typings.MaskArrayType

    # mask is both set in self and returned
    def generate_mask(
            self,
            tensor: torch.Tensor,
            force_recompute: bool = False,
    ) -> enpheeph.utils.typings.MaskArrayType:
        if self.mask is None or force_recompute:
            device_type = tensor.device.type
            conv_fn = self.TORCH_CONVERSION[device_type]
            if conv_fn is None:
                raise RuntimeError(
                        f"The conversion with type {device_type} "
                        "is not supported as the corresponding module is not "
                        "installed"
                )
            # we get the library we are using
            library = self.LIBRARY[device_type]
            
            # NOTE: the following process is used to process the index,
            # based on bitwidth and type
            # the index may start from a non-compatible form, which is then 
            # checked and verified against the PyTorch indexing capabilities
            # we get the dtype to compute its length in bytes, the return 
            # intermediate value is the dimension of the dtype in bytes
            bytewidth = tensor.element_size
            # we create the boolean mask in torch, depending on whether we
            # use 0 or 1 to fill the non-selected values
            bit_mask_info = enpheeph.utils.data_classes.BitFaultMaskInfo.from_bit_fault_value(self.fault_locator.bit_fault_value)
            bool_mask = torch.tensor([bit_mask_info.fill_value] * bytewidth * 8, dtype=torch.bool)
            # we set the selected bits to the value provided by the fault
            # locator
            bool_mask[self.fault_locator.injection_location.bit_index] = bit_mask_info.mask_value
            # we get the correct indices from the boolean mask
            # we convert it to indices in standard Python to create the final
            # integer representation
            indices = torch.where(bool_mask)[0].tolist()
            # we get the final integer representation for the mask
            int_mask = sum(2 ** i for i in indices)
            # placeholder for having device and dtype to be converted
            zero_tensor = torch.zeros(0, device=tensor.device, dtype=tensor.dtype, requires_grad=False)
            # we convert the array, using the correct device for cupy
            if device_type == 'cuda':
                with conv_fn(zero_tensor).device:
                    mask_array = library.array(int_mask, dtype=library.dtype(f'u{str(bytewidth)}'))
            elif device_type == 'cpu':
                mask_array = library.array(int_mask, dtype=library.dtype(f'V{str(bytewidth)}'))
            else:
                raise RuntimeError()
            # we convert it using view
            mask_array = mask_array.view(dtype=conv_fn(zero_tensor).dtype)
            # we create the fill value, using 2 ** bitwidth - 1 to create
            # a all 1 or all 0 array
            fill_value = library.array((2 ** (bytewidth * 8) - 1) * bit_mask_info.fill_value, dtype=library.dtype(f'u{str(bytewidth)}'))
            # we convert it using view
            fill_value = fill_value.view(dtype=dtype)
            # we fill an array with the same shape of the original torch tensor
            # filled with the fill value, and we insert the mask in the correct
            # indices
            mask = library.broadcast_to(fill_value, tensor.shape)
            mask[self.fault_locator.injection_location.tensor_index] = mask_array
            # we get the mask back into pytorch format
            if device_type == 'cpu':
                mask = torch.from_numpy(mask)
            elif device_type == 'cuda':
                mask = torch.utils.dlpack.from_dlpack(mask.toDlpack())
            else:
                raise RuntimeError()
        else:
            mask = self.mask
        self.mask = mask
        return self.mask


    # we return the injected tensor
    def inject_mask(self, tensor):
        if self._mask is None:
            mask = self.generate_mask(tensor)