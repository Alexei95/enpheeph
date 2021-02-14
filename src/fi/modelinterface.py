# summary library https://github.com/sksq96/pytorch-summary

# this model interface is required for providing the correct info to the
# fault injection manager / experiment, such as layers, sizes, etc.
# for PyTorch, use torchsummary, because PyTorchLightning does not have support
# for MACs, memory, etc.
# it also provides the fault injection wrapper to run the fault injections in
# PyTorch Lightning
# FIXME: in future it will be an abstract class providing all the different
# interfaces, such as PyTorch-Lightning, PyTorch, TensorFlow, ...
# FIXME: when providing different interfaces, it should also have a standard
# interface for the different summary classes. For now it is not needed.

import dataclasses

import pytorch_lightning
import torch
import torchsummary  # https://github.com/sksq96/pytorch-summary


@dataclasses.dataclass(init=True)
class ModelInterface(pytorch_lightning.LightningModule):
    model: torch.nn.Module
    summary: torchsummary.summary = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        if self.summary is None:
            self.summary = torchsummary.summary(self._model)

    def add_fault(self, fault):
