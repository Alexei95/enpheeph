import enpheeph.handlers.injectionhandlerabc

import torch


class PyTorchInjectionHandler(
        enpheeph.handlers.injectionhandlerabc.InjectionHandlerABC,
):
    def on_setup_end(self, output, model):
        for i in itertools.chain(self.active_monitors, self.active_faults):
            module = self.get_module(model, i.module_name)
            new_module = i.setup(module)
            self.set_module(model, i.module_name, new_module)
        return model

    def on_teardown_end(self, output, model):
        for i in itertools.chain(self.active_monitors, self.active_faults):
            module = self.get_module(model, i.module_name)
            new_module = i.teardown(module)
            self.set_module(model, i.module_name, new_module)
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
