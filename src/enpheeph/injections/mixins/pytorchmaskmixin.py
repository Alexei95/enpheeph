# -*- coding: utf-8 -*-
import abc
import typing

import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.injections.pytorchinjectionabc
import enpheeph.utils.data_classes
import enpheeph.utils.functions
import enpheeph.utils.imports
import enpheeph.utils.typings

if typing.TYPE_CHECKING or enpheeph.utils.imports.IS_TORCH_AVAILABLE:
    import torch


class PyTorchMaskMixin(abc.ABC):
    # the used variables in the functions, must be initialized properly
    location: enpheeph.utils.data_classes.FaultLocation
    low_level_plugin: (
        # black has issues with long names
        # fmt: off
        enpheeph.injections.plugins.mask.
        lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
        # fmt: on
    )
    mask: typing.Optional["torch.Tensor"]

    # mask is both set in self and returned
    def generate_mask(
        self,
        tensor: "torch.Tensor",
        force_recompute: bool = False,
    ) -> "torch.Tensor":
        if self.mask is None or force_recompute:
            # NOTE: the following process is used to process the index,
            # based on bitwidth and type
            # the index may start from a non-compatible form, which is then
            # checked and verified against the PyTorch indexing capabilities
            # we get the dtype to compute its length in bytes, the return
            # intermediate value is the dimension of the dtype in bytes
            bytewidth = tensor.element_size()
            # we create the boolean mask in torch, depending on whether we
            # use 0 or 1 to fill the non-selected values
            bit_mask_info = (
                enpheeph.utils.data_classes.BitFaultMaskInfo.from_bit_fault_value(
                    self.location.bit_fault_value
                )
            )
            bool_mask: "torch.Tensor" = torch.tensor(
                [bit_mask_info.fill_value] * bytewidth * 8, dtype=torch.bool
            )
            # we set the selected bits to the value provided by the fault
            # locator
            bool_mask[self.location.bit_index] = bit_mask_info.mask_value
            # we get the correct indices from the boolean mask
            # we convert it to indices in standard Python to create the final
            # integer representation
            indices: typing.List[int] = torch.where(bool_mask)[0].tolist()
            # we get the final integer representation for the mask
            int_mask = sum(2 ** i for i in indices)
            # placeholder for having device and dtype to be converted
            tensor_placeholder: "torch.Tensor" = torch.zeros(
                0,
                device=tensor.device,
                dtype=tensor.dtype,
                requires_grad=False,
            )
            # we create the low-level mask
            mask_array = self.low_level_plugin.make_mask_array(
                int_mask,
                self.location.tensor_index,
                (2 ** (bytewidth * 8) - 1) * bit_mask_info.fill_value,
                tensor.shape,
                tensor_placeholder,
            )
            # we convert the mask back to PyTorch
            mask = self.low_level_plugin.to_torch(mask_array)
        else:
            mask = self.mask
        self.mask = mask
        return self.mask

    # we return the injected tensor
    def inject_mask(
        self,
        tensor: "torch.Tensor",
        force_recompute: bool = False,
    ) -> "torch.Tensor":
        self.generate_mask(tensor, force_recompute=force_recompute)

        bit_mask_info = (
            enpheeph.utils.data_classes.BitFaultMaskInfo.from_bit_fault_value(
                self.location.bit_fault_value
            )
        )

        low_level_tensor = self.low_level_plugin.from_torch(tensor)
        # mypy generates an error since self.mask can be None
        # however we call self.generate_mask that will set the mask or raise errors
        # stopping the execution
        low_level_mask = self.low_level_plugin.from_torch(
            # sometimes the following line fails, use type: ignore[arg-type]
            self.mask
        )

        bitwise_tensor = self.low_level_plugin.to_bitwise_type(low_level_tensor)
        bitwise_mask = self.low_level_plugin.to_bitwise_type(low_level_mask)

        bitwise_injected_tensor = bit_mask_info.operation.value(
            bitwise_tensor,
            bitwise_mask,
        )

        low_level_injected_tensor = self.low_level_plugin.to_target_type(
            bitwise_injected_tensor,
            low_level_tensor,
        )

        injected_tensor = self.low_level_plugin.to_torch(low_level_injected_tensor)

        return injected_tensor
