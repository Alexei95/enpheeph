from . import faultinjectorabc


class ActivationFaultInjector(faultinjectorabc.FaultInjectorABC):
    def __init__(self, injection_module, *args, **kwargs):
        super().__init__(*args, injection_module=injection_module, **kwargs)

        self._injection_module = injection_module

    def update_module(self, parent_module, target_module_name):
        # we get the original module and we return 
        module = getattr(parent_module, target_module_name)
        new_module = torch.nn.Sequential(module, self._injection_module)
        return new_module

FAULT_INJECTOR = {ActivationFaultInjector.__name__: ActivationFaultInjector}
