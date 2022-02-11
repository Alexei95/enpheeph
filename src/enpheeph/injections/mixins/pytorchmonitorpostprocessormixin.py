# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import abc
import typing

import enpheeph.utils.classes
import enpheeph.utils.data_classes
import enpheeph.utils.enums
import enpheeph.utils.functions
import enpheeph.utils.imports

if (
    typing.TYPE_CHECKING
    or enpheeph.utils.imports.MODULE_AVAILABILITY[enpheeph.utils.imports.TORCH_NAME]
):
    import torch


def torch_geometric_mean(tensor: "torch.Tensor", dim: int = -1) -> "torch.Tensor":
    log_x: "torch.Tensor" = torch.log(tensor)
    result: "torch.Tensor" = torch.exp(torch.mean(log_x, dim=dim))
    return result


class PyTorchMonitorPostProcessorMixin(abc.ABC):
    enabled_metrics: enpheeph.utils.enums.MonitorMetric
    monitor_location: enpheeph.utils.data_classes.MonitorLocation

    def postprocess(self, tensor: "torch.Tensor") -> typing.Dict[str, typing.Any]:
        dict_ = {}

        skip_if_error = enpheeph.utils.classes.SkipIfErrorContextManager(
            NotImplementedError
        )

        metric_class = self.enabled_metrics.__class__
        if metric_class.StandardDeviation in self.enabled_metrics:
            with skip_if_error:
                dict_[metric_class.StandardDeviation.name] = torch.std(
                    tensor, unbiased=True
                ).item()
        if metric_class.Maximum in self.enabled_metrics:
            with skip_if_error:
                dict_[metric_class.Maximum.name] = torch.max(tensor).item()
        if metric_class.Minimum in self.enabled_metrics:
            with skip_if_error:
                dict_[metric_class.Minimum.name] = torch.min(tensor).item()
        if metric_class.ArithmeticMean in self.enabled_metrics:
            with skip_if_error:
                dict_[metric_class.ArithmeticMean.name] = torch.mean(tensor).item()
        if metric_class.GeometricMean in self.enabled_metrics:
            with skip_if_error:
                dict_[metric_class.GeometricMean.name] = torch_geometric_mean(
                    tensor
                ).item()

        return dict_
