import torch

from . import injectionmoduleabc


class PyTorchActivationInjectionModule(torch.nn.Module, injectionmoduleabc.InjectionModuleABC):
    def forward(self, x, *args, **kwargs):
        modified_tensor = self._operation(index=self._index, tensor=x)

        return modified_tensor


FAULT_INJECTOR = {PyTorchActivationInjectionModule.__name__: PyTorchActivationInjectionModule}
