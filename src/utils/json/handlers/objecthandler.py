import builtins
import importlib
import types
import typing

import src.utils.mixins.dispatcher
import src.utils.mixins.importutils


class ObjectHandler(
    src.utils.mixins.dispatcher.Dispatcher,
    src.utils.mixins.importutils.ImportUtils,
):
    # this is the default that should be used to get the name associated to
    # this decoder/encoder
    OBJECT_HANDLER_DEFAULT_STRING = 'objecthandler'
    OBJECT_HANDLER_KEYS = ['__callable__']
    OBJECT_HANDLER_EXTRA_KEYS = ['__args__', '__kwargs__', '__import__']

    @classmethod
    def get_default_string(cls):
        return cls.OBJECT_HANDLER_DEFAULT_STRING

    @classmethod
    def get_keys(cls):
        return cls.OBJECT_HANDLER_KEYS

    @classmethod
    def get_extra_keys(cls):
        return cls.OBJECT_HANDLER_EXTRA_KEYS

    @classmethod
    def to_json(cls, *args, **kwargs):
        return NotImplemented

    @classmethod
    def from_json(
            cls,
            dict_: typing.Dict[str, typing.Any],
            silent_error: bool = True,
    ) -> typing.Any:
        # here we will have a dict_ that contains at least __custom__: True
        # and __custom_decoder__
        # hence, we don't need to check them again, only work on the
        # remaining parameters
        # this handler supports
        # __callable__ to mention a callable, which can also be a class
        # __args__ to mention a list of args
        # __kwargs__ to mention a dict of args
        # if __import__ is True, then the callable must be imported
        # if not given or False, then we assume the callable is already present
        # in the namespace
        # we check whether the necessary keys are present in the dict keys,
        # by using the issubset method from the set
        # if not satisfied, we raise ValueError
        if not set(cls.get_keys()).issubset(dict_.keys()):
            # if silent_error is True, then we return the dict without
            # raising the error
            if silent_error:
                return dict_
            else:
                raise ValueError(
                        'The following keys must be '
                        'present to be correctly '
                        'parsed: {}'.format(cls.get_keys())
                )

        callable_str = dict_['__callable__']
        args = dict_.get('__args__', tuple())
        kwargs = dict_.get('__kwargs__', {})
        import_ = dict_.get('__import__', False)

        callable_func = cls.get_callable(
                element=callable_str,
                import_=import_
        )

        return callable_func(*args, **kwargs)
