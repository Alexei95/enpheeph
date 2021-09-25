import abc

import torch

import enpheeph.injections.injectionabc


# only a stub as middle ground
class PyTorchInjectionABC(
        enpheeph.injections.injectionabc.InjectionABC
):
    @property
    @abc.abstractmethod
    def module_name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def register_hook(self, module: torch.nn.Module, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def deregister_hook(self, module: torch.nn.Module, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def save_handle(
            self,
            module: torch.nn.Module,
            handle,
            *args,
            **kwargs
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def gather_handle(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs
    ):  # handle type from PyTorch
        raise NotImplementedError

    @abc.abstractmethod
    def on_setup_start(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs
    ) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def on_setup_end(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs
    ) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def on_teardown_start(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs
    ) -> torch.nn.Module:
        raise NotImplementedError

    @abc.abstractmethod
    def on_teardown_end(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs
    ) -> torch.nn.Module:
        raise NotImplementedError

    def setup(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs
    ) -> torch.nn.Module:
        module = self.on_setup_start(module, *args, **kwargs)

        module, handle = self.register_hook(module, *args, **kwargs)
        self.save_handle(module, handle, *args, **kwargs)

        module = self.on_setup_end(module, *args, **kwargs)

        return module

    def teardown(
            self,
            module: torch.nn.Module,
            *args,
            **kwargs
    ) -> torch.nn.Module:
        module = self.on_teardown_start(module, *args, **kwargs)

        handle = self.gather_handle(module, *args, *kwargs)
        module = self.deregister_hook(module, handle, *args, **kwargs)

        module = self.on_teardown_end(module, *args, **kwargs)

        return module

        