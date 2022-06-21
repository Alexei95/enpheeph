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

import typing

import enpheeph.handlers.plugins.libraryhandlerpluginabc
import enpheeph.injections.abc.injectionabc
import enpheeph.utils.typings

# we plac it after so flake8 does not complain about not-at-the-top imports
if typing.TYPE_CHECKING:
    import torch


class PyTorchHandlerPlugin(
    (enpheeph.handlers.plugins.libraryhandlerpluginabc.LibraryHandlerPluginABC),
):
    def library_setup(
        self,
        model: enpheeph.utils.typings.ModelType,
        active_injections: typing.List[
            enpheeph.injections.abc.injectionabc.InjectionABC
        ],
    ) -> enpheeph.utils.typings.ModelType:
        for inj in active_injections:
            module = self.get_module(model, inj.location.module_name)
            new_module = inj.setup(module)
            self.set_module(model, inj.location.module_name, new_module)
        return model

    def library_teardown(
        self,
        model: enpheeph.utils.typings.ModelType,
        active_injections: typing.List[
            enpheeph.injections.abc.injectionabc.InjectionABC
        ],
    ) -> enpheeph.utils.typings.ModelType:
        for inj in active_injections:
            module = self.get_module(model, inj.location.module_name)
            new_module = inj.teardown(module)
            self.set_module(model, inj.location.module_name, new_module)
        return model

    def get_module(
        self, model: "torch.nn.Module", full_module_name: str
    ) -> "torch.nn.Module":
        dest_module = model
        for submodule in full_module_name.split("."):
            dest_module = getattr(dest_module, submodule)
        return dest_module

    def set_module(
        self,
        model: "torch.nn.Module",
        full_module_name: str,
        module: "torch.nn.Module",
    ) -> None:
        dest_module = model
        module_names_split = full_module_name.split(".")
        module_names = module_names_split[:-1]
        target_module_name = module_names_split[-1]
        for submodule in module_names:
            dest_module = getattr(dest_module, submodule)
        setattr(dest_module, target_module_name, module)
