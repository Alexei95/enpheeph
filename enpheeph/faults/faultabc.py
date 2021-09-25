import abc

import enpheeph.injections.injectionabc


class FaultABC(enpheeph.injections.injectionabc.InjectionABC):
    def __init__(self, fault_locator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fault_locator = fault_locator
