import enpheeph.handlers.generalpurposeinjectionhandlerabc


class PyTorchInjectionHandler(
        enpheeph.handlers.generalpurposeinjectionhandlerabc.
        GeneralPurposeInjectionHandlerABC,
):
    def get_module(self, model: 'torch.nn.Module', full_module_name: str):
        dest_module = model
        for submodule in full_module_name.split('.'):
            dest_module = getattr(dest_module, submodule)
        return dest_module

    def set_module(
            self,
            model: 'torch.nn.Module',
            full_module_name: str,
            module: 'torch.nn.Module'
    ):
        dest_module = model
        module_names_split = full_module_name.split('.')
        module_names = module_names_split[:-1]
        target_module_name = module_names_split[-1]
        for submodule in module_names:
            dest_module = getattr(dest_module, submodule)
        setattr(dest_module, target_module_name, module)
