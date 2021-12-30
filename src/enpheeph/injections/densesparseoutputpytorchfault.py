# -*- coding: utf-8 -*-
import typing

import enpheeph.injections.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmaskmixin
import enpheeph.injections.mixins.pytorchsparseinterfacemixin
import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.utils.data_classes

# we move this import down
if typing.TYPE_CHECKING:
    import torch


class DenseSparseOutputPyTorchFault(
    enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC,
    enpheeph.injections.mixins.pytorchmaskmixin.PyTorchMaskMixin,
    enpheeph.injections.mixins.pytorchsparseinterfacemixin.PyTorchSparseInterfaceMixin,
):
    location: enpheeph.utils.data_classes.FaultLocation
    low_level_plugin: (
        # black has issues with long names
        # fmt: off
        enpheeph.injections.plugins.mask.
        lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
        # fmt: on
    )
    mask: typing.Optional["torch.Tensor"]

    def __init__(
        self,
        location: enpheeph.utils.data_classes.FaultLocation,
        low_level_torch_plugin: (
            # black has issues with long names
            # fmt: off
            enpheeph.injections.plugins.mask.
            lowleveltorchmaskpluginabc.LowLevelTorchMaskPluginABC
            # fmt: on
        ),
    ) -> None:
        super().__init__()

        self.location = location
        self.low_level_plugin = low_level_torch_plugin

        self.handle = None
        self.mask = None

    @property
    def module_name(self) -> str:
        return self.location.module_name

    def output_fault_hook(
        self,
        module: "torch.nn.Module",
        input: typing.Union[typing.Tuple["torch.Tensor"], "torch.Tensor"],
        output: "torch.Tensor",
    ) -> None:
        target = self.get_sparse_injection_parameter(output)

        # using this indexing we cover only the main dimensions, as it is difficult
        # to select a specific batch on the sparse coordinates
        # sparse coordinates are [number of dimensions, number of elements]
        self.indexing_plugin.select_active_dimensions(
            [
                enpheeph.utils.enums.DimensionType.Tensor,
            ],
            autoshift_to_boundaries=True,
            fill_empty_index=False,
        )
        # we don't need to return the tensors as all the changes are in-place
        self.generate_mask(target, tensor_only=None)

        self.inject_mask(target, tensor_only=None)

        # we reset the active dimensions
        self.indexing_plugin.reset_active_dimensions()

    def setup(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        self.handle = module.register_forward_hook(self.output_fault_hook)

        return module
