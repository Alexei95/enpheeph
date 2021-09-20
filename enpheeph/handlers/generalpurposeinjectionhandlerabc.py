import abc
import itertools

import enpheeph.handlers.injectionhandlerabc


class GeneralPurposeInjectionHandlerABC(
        enpheeph.handlers.injectionhandlerabc.InjectionHandlerABC,
):
    def setup(self, model, *args, **kwargs):
        super().setup(*args, **kwargs)

        for i in itertools.chain(self.active_monitors, self.active_faults):
            module = self.get_module(model, i.module_name)
            new_module = i.setup(module)
            self.set_module(model, i.module_name, new_module)

    def teardown(self, model, *args, **kwargs):
        super().teardown(*args, **kwargs)

        for i in itertools.chain(self.active_monitors, self.active_faults):
            module = self.get_module(model, i.module_name)
            new_module = i.teardown(module)
            self.set_module(model, i.module_name, new_module)

    @abc.abstractmethod
    def get_module(self, model, full_module_name):
        return NotImplemented

    @abc.abstractmethod
    def set_module(self, model, full_module_name, module):
        return NotImplemented
