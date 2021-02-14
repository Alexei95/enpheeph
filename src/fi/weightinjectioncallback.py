import torch

from . import basefaultdescriptor
from . import baseinjectioncallback


# this class covers injections in weights
class WeightInjectionCallback(baseinjectioncallback.BaseInjectionCallback):
    def __post_init__(self):
        # we check all the faults to be injected are weight faults
        for fault in self.fault_descriptor_list:
            assert fault.type == basefaultdescriptor.ParameterType.Weight

    def init_module(self, fault: basefaultdescriptor.BaseFaultDescriptor,
                    module: torch.nn.Module):
        pass
