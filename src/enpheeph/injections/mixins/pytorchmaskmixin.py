# -*- coding: utf-8 -*-
import abc
import typing

import enpheeph.injections.plugins.indexing.indexingpluginabc
import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.injections.pytorchinjectionabc
import enpheeph.utils.data_classes
import enpheeph.utils.functions
import enpheeph.utils.imports
import enpheeph.utils.typings

if (
    typing.TYPE_CHECKING
    or enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.TORCH_NAME]
):
    import torch


class PyTorchMaskMixin(abc.ABC):
    # we need the index plugin to simplify the handling of the indices
    indexing_plugin: (
        enpheeph.injections.plugins.indexing.indexingpluginabc.IndexingPluginABC
    )
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

    def set_tensor_only_indexing(self) -> None:
        self.indexing_plugin.select_active_dimensions(
            [
                enpheeph.utils.enums.DimensionType.Tensor,
            ],
            autoshift_to_boundaries=False,
            fill_empty_index=True,
            filler=slice(None, None),
        )

    def set_batch_tensor_indexing(self) -> None:
        self.indexing_plugin.select_active_dimensions(
            [
                enpheeph.utils.enums.DimensionType.Batch,
                enpheeph.utils.enums.DimensionType.Tensor,
            ],
            autoshift_to_boundaries=False,
            fill_empty_index=True,
            filler=slice(None, None),
        )

    # mask is both set in self and returned
    def generate_mask(
        self,
        tensor: "torch.Tensor",
        force_recompute: bool = False,
        # if True we use set_tensor_only_indexing, if False we use
        # set_batch_tensor_indexing
        # if explicitly non-boolean, we skip it, to allow for custom configurations
        tensor_only: typing.Optional[bool] = True,
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
            # we set up the indices depending on the flag
            # if the flag is different, we leave the existing active dimensions
            if tensor_only is True:
                self.set_tensor_only_indexing()
            elif tensor_only is False:
                self.set_batch_tensor_indexing()
            tensor_shape = self.indexing_plugin.filter_dimensions(
                tensor.shape,
            )
            # we create the low-level mask
            # using the filtered dimensions
            # we only need the tensor_index, as we do not cover the time/batch
            # dimensions
            mask_array = self.low_level_plugin.make_mask_array(
                int_mask=int_mask,
                # we use only the tensor index as the mask will be the same even
                # across different batches/time-steps
                # so it can be expanded/repeated later
                mask_index=self.location.dimension_index[
                    enpheeph.utils.enums.DimensionType.Tensor
                ],
                int_fill_value=(2 ** (bytewidth * 8) - 1) * bit_mask_info.fill_value,
                shape=tensor_shape,
                torch_placeholder=tensor_placeholder,
            )
            # we convert the mask back to PyTorch
            mask = self.low_level_plugin.to_torch(mask_array)

            # the indices are reset if we have set them up ourselvels
            if isinstance(tensor_only, bool):
                self.indexing_plugin.reset_active_dimensions()
        else:
            mask = self.mask

        self.mask = mask

        return self.mask

    # we return the injected tensor
    def inject_mask(
        self,
        tensor: "torch.Tensor",
        # if True we use set_tensor_only_indexing, if False we use
        # set_batch_tensor_indexing
        # if explicitly non-boolean, we skip it, to allow for custom configurations
        tensor_only: typing.Optional[bool] = True,
    ) -> "torch.Tensor":
        if self.mask is None:
            raise RuntimeError("Please call generate_mask before injection")

        bit_mask_info = (
            enpheeph.utils.data_classes.BitFaultMaskInfo.from_bit_fault_value(
                self.location.bit_fault_value
            )
        )
        # we set up the indices depending on the flag
        if tensor_only is True:
            self.set_tensor_only_indexing()
        elif tensor_only is False:
            self.set_batch_tensor_indexing()

        selected_batches_tensor = tensor[
            self.indexing_plugin.join_indices(
                {
                    **self.location.dimension_index,
                    **{
                        enpheeph.utils.enums.DimensionType.Tensor: ...,
                    },
                },
            )
        ]

        low_level_tensor = self.low_level_plugin.from_torch(
            selected_batches_tensor,
        )
        # mypy generates an error since self.mask can be None
        # however we call self.generate_mask that will set the mask or raise errors
        # stopping the execution
        low_level_mask = self.low_level_plugin.from_torch(
            # we use expand as to expand the mask onto the selected batches
            # dimension
            # expand creates views, so we should not change the elements in place,
            # but it is doable as we are working on the mask which will not be modified
            # sometimes the following line fails with mypy, use type: ignore[arg-type]
            self.mask.expand_as(selected_batches_tensor)
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

        final_injected_tensor = injected_tensor[
            self.indexing_plugin.join_indices(
                {
                    **self.location.dimension_index,
                    **{
                        enpheeph.utils.enums.DimensionType.Tensor: ...,
                    },
                },
            )
        ]

        # the indices are reset if we have set them up ourselvels
        if isinstance(tensor_only, bool):
            self.indexing_plugin.reset_active_dimensions()

        return final_injected_tensor
