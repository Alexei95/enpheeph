import abc


class ModelWrapperABC(abc.ABC):
    def __init__(self, model, *args, **kwargs):
        self._model = model

    @abc.abstractmethod
    def set_module(self, name, value, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_module(self, name, *args, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def copy_module(module, *args, **kwargs):
        pass

    def has_module(self, name, *args, **kwargs):
        try:
            self.get_module(name)
        except AttributeError:
            return False
        else:
            return True

    # we implement a custom getattribute to have direct access to model
    # attributes, by prepending 'model_' before the attribute
    def __getattribute__(self, name: str):
        if name.startswith('model_'):
            return getattr(self._model, name)
        return super().__getattribute__(name)
