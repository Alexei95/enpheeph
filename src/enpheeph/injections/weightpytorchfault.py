# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import copy
import typing

import enpheeph.injections.abc.faultabc
import enpheeph.injections.abc.pytorchinjectionabc
import enpheeph.injections.mixins.pytorchmaskmixin
import enpheeph.injections.mixins.pytorchtensorobjectvalidatormixin
import enpheeph.injections.plugins.mask.abc.lowleveltorchmaskpluginabc
import enpheeph.utils.data_classes

# we move this import down
if typing.TYPE_CHECKING:
    import torch


# no need to use handles here as the change is done when the injection is setup
class WeightPyTorchFault(
    enpheeph.injections.abc.faultabc.FaultABC,
    enpheeph.injections.abc.pytorchinjectionabc.PyTorchInjectionABC,
    enpheeph.injections.mixins.pytorchmaskmixin.PyTorchMaskMixin,
    (
        # fmt: off
        enpheeph.injections.mixins.
        pytorchtensorobjectvalidatormixin.PyTorchTensorObjectValidatorMixin
        # fmt: on
    ),
):
    backup: typing.Optional["torch.Tensor"]
    # we need the index plugin to simplify the handling of the indices
    indexing_plugin: (
        enpheeph.injections.plugins.indexing.abc.indexingpluginabc.IndexingPluginABC
    )
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
        indexing_plugin: enpheeph.injections.plugins.indexing.abc.indexingpluginabc,
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

        self.indexing_plugin = indexing_plugin
        self.location = location
        self.low_level_plugin = low_level_torch_plugin

        self.backup = None
        self.handle = None
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
            # sometimes type: ignore[arg-type] might be required for the following line
            # mypy gives error as parameter_name can be None, but it cannot be since
            # the dataclass checks for the validity
            # so we simply cast it here
            typing.cast(str, self.location.parameter_name),
        )
        self.backup = copy.deepcopy(weight)

        self.generate_mask(
            weight,
            tensor_only=True,
            batches_exist=False,
        )
        masked_weight = self.inject_mask(
            weight,
            tensor_only=True,
            batches_exist=False,
        )

        setattr(
            module,
            # sometimes type: ignore[arg-type] might be required for the following line
            # mypy gives error as parameter_name can be None, but it cannot be since
            # the dataclass checks for the validity
            # so we simply cast it here
            typing.cast(str, self.location.parameter_name),
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
            typing.cast(str, self.location.parameter_name),
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
