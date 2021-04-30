import torch

import src.fi.injection.injectioncallback
import src.fi.utils.enums.parametertype


# we map the class to the activation injection
@src.fi.injection.injectioncallback.InjectionCallback.register_decorator(
        src.fi.utils.enums.parametertype.ParameterType.Activation
)
# NOTE: we can only have one of the following module per layer, as the parsing
# of the top module is done statically on the original structure, not on the
# updated layers
# NOTE: we cannot have dataclasses go with torch.nn.Module as Module.__init__
# must be called before the dataclasses init
# FIXME: implement also backward for fault-aware training
class ActivationInjectionModule(torch.nn.Module):
    def __init__(
                self,
                fault: 'src.fi.injection.faultdescriptor.FaultDescriptor',
                module: 'torch.nn.Module'

    ):
        super().__init__()

        self.fault = fault
        self.module = module

    def forward(self, x):
        # we get the exact result from the previous module
        y_temp = self.module(x)
        # we inject the faults in the tensor
        y = src.fi.fiutils.inject_tensor_fault_pytorch(
                tensor=y_temp,
                fault=self.fault,
        )
        # we return the fault-injected tensor
        return y
