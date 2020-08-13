from . import faultinjectorabc


class ActivationFaultInjector(faultinjectorabc.FaultInjectorABC):
    def __init__(self, sampler, injector, *args, **kwargs):
        super().__init__(*args, sampler=sampler, injector=injector, **kwargs)

        self._sampler = sampler
        self._injector = injector

    def update_module(self, parent_module, target_module_name):
        #
        module = getattr(parent_module, target_module_name)
        sample =
