import abc


class CallbackInterfaceABC(abc.ABC):
    def __init__(self, fault_injector_manager, wrapper_cls, *args, **kwargs):
        self._fault_injector_manager = fault_injector_manager
        self._wrapper_cls = wrapper_cls

    def setup(self, model):
        self._fault_injector_manager.setup_fi(self._wrapper_cls(model))

    def restore(self, model):
        self._fault_injector_manager.restore_fi(self._wrapper_cls(model))
