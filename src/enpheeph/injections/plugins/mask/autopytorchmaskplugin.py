# -*- coding: utf-8 -*-
import typing

import enpheeph.injections.plugins.mask
import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.utils.functions
import enpheeph.utils.imports

if typing.TYPE_CHECKING:
    import torch

    import enpheeph.injections.plugins.mask.numpypytorchmaskplugin
    import enpheeph.injections.plugins.mask.cupypytorchmaskplugin
else:
    if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.TORCH_NAME]:
        import torch
    if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.CUPY_NAME]:
        import enpheeph.injections.plugins.mask.cupypytorchmaskplugin
    if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.NUMPY_NAME]:
        import enpheeph.injections.plugins.mask.numpypytorchmaskplugin


class AutoPyTorchMaskPlugin(
    # we disable black to avoid too long line issue in flake8
    # fmt: off
    (
        enpheeph.injections.plugins.mask.
        lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
    ),
    # fmt: on
):
    CPU_TORCH_DEVICE = "cpu"
    GPU_TORCH_DEVICE = "cuda"

    FROM_TORCH = {
        CPU_TORCH_DEVICE: enpheeph.injections.plugins.mask.NumPyPyTorchMaskPlugin()
        if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.NUMPY_NAME]
        else None,
        GPU_TORCH_DEVICE: enpheeph.injections.plugins.mask.CuPyPyTorchMaskPlugin()
        if enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.CUPY_NAME]
        else None,
    }
    TO_TORCH = {
        enpheeph.utils.imports.CUPY_NAME: FROM_TORCH[GPU_TORCH_DEVICE],
        enpheeph.utils.imports.NUMPY_NAME: FROM_TORCH[CPU_TORCH_DEVICE],
    }

    def _get_from_torch_plugin_instance(
        self, tensor: "torch.Tensor"
    ) -> enpheeph.injections.plugins.mask.LowLevelTorchMaskPluginABC:
        return self.FROM_TORCH[tensor.device.type]

    def _get_to_torch_plugin_instance(
        self,
        array: enpheeph.utils.typings.ArrayType,
    ) -> enpheeph.injections.plugins.mask.LowLevelTorchMaskPluginABC:
        return self.TO_TORCH[
            typing.cast(
                str,
                enpheeph.utils.functions.get_object_library(array),
            )
        ]

    def to_torch(self, array: enpheeph.utils.typings.ArrayType) -> "torch.Tensor":
        return typing.cast(
            "torch.Tensor", self._get_to_torch_plugin_instance(array).to_torch(array)
        )

    def from_torch(self, tensor: "torch.Tensor") -> enpheeph.utils.typings.ArrayType:
        return self._get_from_torch_plugin_instance(tensor).from_torch(tensor)

    def to_bitwise_type(
        self, array: enpheeph.utils.typings.ArrayType
    ) -> enpheeph.utils.typings.ArrayType:
        return self._get_to_torch_plugin_instance(array).to_bitwise_type(array)

    def to_target_type(
        self,
        array: enpheeph.utils.typings.ArrayType,
        target: enpheeph.utils.typings.ArrayType,
    ) -> enpheeph.utils.typings.ArrayType:
        return self._get_to_torch_plugin_instance(target).to_target_type(array, target)

    def make_mask_array(
        self,
        int_mask: int,
        mask_index: enpheeph.utils.typings.AnyIndexType,
        int_fill_value: int,
        shape: typing.Sequence[int],
        torch_placeholder: "torch.Tensor",
    ) -> enpheeph.utils.typings.ArrayType:
        return self._get_from_torch_plugin_instance(torch_placeholder).make_mask_array(
            int_mask=int_mask,
            mask_index=mask_index,
            int_fill_value=int_fill_value,
            shape=shape,
            torch_placeholder=torch_placeholder,
        )
