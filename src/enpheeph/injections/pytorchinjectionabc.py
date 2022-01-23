# -*- coding: utf-8 -*-
import abc
import typing

import enpheeph.injections.injectionabc

# to avoid flake complaining that imports are after if, even though torch is 3rd-party
# library so it should be before self-imports
if typing.TYPE_CHECKING:
    import torch


class PyTorchInjectionABC(enpheeph.injections.injectionabc.InjectionABC):
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
