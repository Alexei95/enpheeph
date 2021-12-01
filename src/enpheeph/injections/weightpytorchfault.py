# -*- coding: utf-8 -*-
import copy
import typing

import enpheeph.injections.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmaskmixin
import enpheeph.injections.plugins.mask.lowleveltorchmaskpluginabc
import enpheeph.utils.data_classes

# we move this import down
if typing.TYPE_CHECKING:
    import torch


class WeightPyTorchFault(
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

        self.backup = None
        self.mask = None

    @property
    def module_name(self) -> str:
        return self.location.module_name

    def inject_weight(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        if self.backup is not None:
            raise ValueError(
                "This method must be called only when setting up the injection"
            )

        weight = getattr(
            module,
            self.location.parameter_name,  # type: ignore[arg-type]
        )
        self.backup = copy.deepcopy(weight)
        self.generate_mask(weight)
        masked_weight = self.inject_mask(weight)

        setattr(
            module,
            self.location.parameter_name,  # type: ignore[arg-type]
            masked_weight,
        )
        return module

    def restore_weight(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        if self.backup is None:
            raise ValueError(
                "This method must be called only when tearing down the injection"
            )

        setattr(  # type: ignore[unreachable]
            module,
            self.location.parameter_name,
            copy.deepcopy(self.backup),
        )
        self.backup = None

        return module

    def setup(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        module = self.inject_weight(module)

        return module

    # we need to override the teardown as it is not common to the normal hook
    # teardowns
    def teardown(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        module = self.restore_weight(module)

        return module
