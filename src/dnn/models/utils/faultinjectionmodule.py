from .. import moduleabc


class FaultInjectionModule(moduleabc.ModuleABC):
    def __init__(self,
                       *args, **kwargs):
        kwargs.update(
                        aaa
                     )

        super().__init__(*args, **kwargs)

    def add_fault(self, *args, **kwargs):
        pass

    def remove_fault(self, *args, **kwargs):
        pass

    def reset_faults(self, *args, **kwargs):
        pass
