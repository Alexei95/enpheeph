import abc
import typing

import torch

import enpheeph.injections.injectionabc


class PyTorchInjectionABC(enpheeph.injections.injectionabc.InjectionABC):
    @abc.abstractmethod
    def setup(
            self,
            module: torch.nn.Module,
    ) -> torch.nn.Module:
        return NotImplemented

    @abc.abstractmethod
    def teardown(
            self,
            module: torch.nn.Module,
    ) -> torch.nn.Module:
        return NotImplemented

    @property
    @abc.abstractmethod
    def module_name(self) -> str:
        return NotImplemented

