import abc
import collections.abc

import torch
import torch.quantization
import pytorch_lightning as pl

from .. import moduleabc


# the following steps are only for preparing the module to the static
# quantization: https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
# TODO: complete implementation, missing fuse_model, torch.quantization.default_qconfig
#       and torch.quantization.prepare, torch.quantization.convert
# TODO: cross-check with quantization-aware training
class StaticQuantizationModule(moduleabc.ModuleABC):
    def __init__(self, quantization_ready=True, *args, **kwargs):
        '''Here we save all the useful settings, like a loss and an accuracy
        functions, accepting predictions and targets in this order.'''
        kwargs.update({'quantization_ready': quantization_ready})

        super().__init__(*args, **kwargs)

        self._quantization_ready = quantization_ready
        self._quantization_forward_pre_hook_handle = None
        self._quantization_forward_pre_hook_handle = None
        self.torch_add = torch.add

        if self._quantization_ready:
            self.enable_quantization()

    def enable_quantization(self):
        if self._quantization_forward_pre_hook_handle is None and self._quantization_forward_pre_hook_handle is None:
            self._quantization_forward_pre_hook_handle = self.register_forward_pre_hook(self.quantization_forward_pre_hook)
            self._quantization_forward_pre_hook_handle = self.register_forward_hook(self.quantization_forward_post_hook)

        self.torch_add = torch.nn.quantized.FloatFunctional()

    def disable_quantization(self):
        if self._quantization_forward_pre_hook_handle is not None:
            self._quantization_forward_pre_hook_handle.remove()
            self._quantization_forward_pre_hook_handle = None

        if self._quantization_forward_pre_hook_handle is not None:
            self._quantization_forward_pre_hook_handle.remove()
            self._quantization_forward_pre_hook_handle = None

        self.torch_add = torch.add

    @staticmethod
    # this hook is called before the forward to add the quantization stub to
    # the input
    def quantization_forward_pre_hook(self, input):
        return torch.quantization.QuantStub()(input)

    @staticmethod
    # this hook is called after the forward to add the quantization destub to
    # the input
    def quantization_forward_post_hook(self, input, output):
        return torch.quantization.DeQuantStub()(output)
