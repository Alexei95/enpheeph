import typing

import torch

import enpheeph.handlers.plugins.libraryhandlerpluginabc
import enpheeph.injections.injectionabc
import enpheeph.utils.typings


class PyTorchHandlerPlugin(
        (
                enpheeph.handlers.plugins.libraryhandlerpluginabc.
                LibraryHandlerPluginABC
        ),
):
    def library_setup(
            self,
            model: enpheeph.utils.typings.ModelType,
            active_injections: typing.List[
                    enpheeph.injections.injectionabc.InjectionABC
            ],
    ) -> enpheeph.utils.typings.ModelType:
        for inj in active_injections:
            module = self.get_module(model, inj.module_name)
            new_module = inj.setup(module)
            self.set_module(model, inj.module_name, new_module)
        return model

    def library_teardown(
            self,
            model: enpheeph.utils.typings.ModelType,
            active_injections: typing.List[
                    enpheeph.injections.injectionabc.InjectionABC
            ],
    ) -> enpheeph.utils.typings.ModelType:
        for inj in active_injections:
            module = self.get_module(model, inj.module_name)
            new_module = inj.teardown(module)
            self.set_module(model, inj.module_name, new_module)
        return model
        
    def get_module(self, model: torch.nn.Module, full_module_name: str):
        dest_module = model
        for submodule in full_module_name.split('.'):
            dest_module = getattr(dest_module, submodule)
        return dest_module

    def set_module(
            self,
            model: torch.nn.Module,
            full_module_name: str,
            module: torch.nn.Module
    ):
        dest_module = model
        module_names_split = full_module_name.split('.')
        module_names = module_names_split[:-1]
        target_module_name = module_names_split[-1]
        for submodule in module_names:
            dest_module = getattr(dest_module, submodule)
        setattr(dest_module, target_module_name, module)
