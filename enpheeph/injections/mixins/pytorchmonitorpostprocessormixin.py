import abc
import typing

import torch

import enpheeph.utils.data_classes
import enpheeph.utils.enums


def torch_geometric_mean(tensor: torch.Tensor) -> torch.Tensor:
    a


class PyTorchMonitorPostProcessorMixIn(abc.ABC):
    enabled_metrics: enpheeph.utils.enums.MonitorMetric
    monitor_location: enpheeph.utils.data_classes.InjectionLocation

    def postprocess(
            self,
            tensor: torch.Tensor
    ) -> typing.Dict[str, typing.Any]:
        dict_ = {}
        
        metric_class = self.enabled_metrics.__class__
        if metric_class.StandardDeviation in self.enabled_metrics:
            dict_[metric_class.StandardDeviation.name] = (
                    torch.std(tensor, unbiased=True).item()
            )
        if metric_class.Maximum in self.enabled_metrics:
            dict_[metric_class.Maximum.name] = (
                    torch.max(tensor).item()
            )
        if metric_class.Minimum in self.enabled_metrics:
            dict_[metric_class.Minimum.name] = (
                    torch.min(tensor).item()
            )
        if metric_class.ArithmeticMean in self.enabled_metrics:
            dict_[metric_class.ArithmeticMean.name] = (
                    torch.mean(tensor).item()
            )
        if metric_class.GeometricMean in self.enabled_metrics:
            pass
        
        return dict_