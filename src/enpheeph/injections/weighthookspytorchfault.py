# -*- coding: utf-8 -*-
import typing

import enpheeph.injections.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmaskmixin
import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.utils.data_classes

# we move this import down
if typing.TYPE_CHECKING:
    import torch

# WARNING: WORK IN PROGRESS, NOT WORKING YET


class WeightHooksPyTorchFault(
    enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC,
    enpheeph.injections.mixins.pytorchmaskmixin.PyTorchMaskMixin,
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

    def weight_fault_pre_hook(
        self,
        module: "torch.nn.Module",
        input: typing.Union[typing.Tuple["torch.Tensor"], "torch.Tensor"],
        output: "torch.Tensor",
    ) -> "torch.Tensor":
        self.generate_mask(output)

        masked_output = self.inject_mask(output)

        return masked_output

    def setup(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        self.pre_handle = module.register_forward_pre_hook(self.weight_fault_pre_hook)
        self.after_handle = module.register_forward_

        return module
