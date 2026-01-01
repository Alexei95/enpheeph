# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2026 Alessio "Alexei95" Colucci
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

import abc
import typing

import enpheeph.injections.abc.injectionabc

# to avoid flake complaining that imports are after if, even though torch is 3rd-party
# library so it should be before self-imports
if typing.TYPE_CHECKING:
    import torch


class PyTorchInjectionABC(enpheeph.injections.abc.injectionabc.InjectionABC):
    handle: typing.Optional["torch.utils.hooks.RemovableHandle"]

    @abc.abstractmethod
    def setup(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        pass

    # we define here the teardown as it should be common for all injections
    # if some injections require particular care, it should be overridden, as long as
    # the signature is the same
    def teardown(
        self,
        module: "torch.nn.Module",
    ) -> "torch.nn.Module":
        # safe get the handle attribute if not defined
        if getattr(self, "handle", None) is not None:
            typing.cast(
                "torch.utils.hooks.RemovableHandle",
                self.handle,
            ).remove()
            self.handle = None

        return module

    @property
    @abc.abstractmethod
    def module_name(self) -> str:
        pass
