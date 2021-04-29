import torch

from . import faultdescriptor
from . import injectioncallback


# this class covers injections in weights
class WeightInjectionCallback(injectioncallback.InjectionCallback):
    def __post_init__(self):
        # we check all the faults to be injected are weight faults
        for fault in self.fault_descriptor_list:
            assert fault.type == faultdescriptor.ParameterType.Weight

    def init_module(self, fault: faultdescriptor.FaultDescriptor,
                    module: torch.nn.Module):
        pass
