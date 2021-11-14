# -*- coding: utf-8 -*-
import abc
import typing

import enpheeph.utils.enums
import enpheeph.utils.typings

# while 3rd party library should be before custom libraries, we move it down to avoid
# flake8 complaining, since it is a conditional import
if typing.TYPE_CHECKING:
    import torch


class LowLevelTorchMaskPluginABC(abc.ABC):
    @abc.abstractmethod
    def to_torch(
        self, array: enpheeph.utils.typings.LowLevelMaskArrayType
    ) -> "torch.Tensor":
        pass

    @abc.abstractmethod
    def from_torch(
        self, tensor: "torch.Tensor"
    ) -> enpheeph.utils.typings.LowLevelMaskArrayType:
        pass

    @abc.abstractmethod
    def to_bitwise_type(
        self, array: enpheeph.utils.typings.LowLevelMaskArrayType
    ) -> enpheeph.utils.typings.LowLevelMaskArrayType:
        pass

    @abc.abstractmethod
    def to_target_type(
        self,
        array: enpheeph.utils.typings.LowLevelMaskArrayType,
        target: enpheeph.utils.typings.LowLevelMaskArrayType,
    ) -> enpheeph.utils.typings.LowLevelMaskArrayType:
        pass

    @abc.abstractmethod
    def make_mask_array(
        self,
        int_mask: int,
        mask_index: enpheeph.utils.typings.IndexMultiDType,
        # this fill value is already final, as is the int mask
        int_fill_value: int,
        shape: typing.Sequence[int],
        torch_placeholder: "torch.Tensor",
    ) -> enpheeph.utils.typings.LowLevelMaskArrayType:
        pass
