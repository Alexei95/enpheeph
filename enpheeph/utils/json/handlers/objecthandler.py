import builtins
import importlib
import types
import typing

import enpheeph.utils.instance_or_classmethod
import enpheeph.utils.mixins.dispatcher
import enpheeph.utils.mixins.importutils


class ObjectHandler(
    # to dispatch the operation to other object handlers
    enpheeph.utils.mixins.dispatcher.Dispatcher,
    enpheeph.utils.mixins.importutils.ImportUtils,
):
    # this is the default that should be used to get the name associated to
    # this decoder/encoder
    OBJECT_HANDLER_DEFAULT_STRING = 'object'
    OBJECT_HANDLER_KEYS = ['__object__']
    OBJECT_HANDLER_EXTRA_KEYS = ['__import__']

    @enpheeph.utils.instance_or_classmethod.instance_or_classmethod
    def get_default_string(self_or_cls):
        return self_or_cls.OBJECT_HANDLER_DEFAULT_STRING

    @enpheeph.utils.instance_or_classmethod.instance_or_classmethod
    def get_keys(self_or_cls):
        return self_or_cls.OBJECT_HANDLER_KEYS

    @enpheeph.utils.instance_or_classmethod.instance_or_classmethod
    def get_extra_keys(self_or_cls):
        return self_or_cls.OBJECT_HANDLER_EXTRA_KEYS

    @enpheeph.utils.instance_or_classmethod.instance_or_classmethod
    def decode_json(
            self_or_cls,
            dict_: typing.Dict[str, typing.Any],
            silent_error: bool = True,
    ) -> typing.Any:
        # here we will have a dict_ that contains at least __custom__: True
        # and __custom_decoder__
        # hence, we don't need to check them again, only work on the
        # remaining parameters
        # this handler supports
        # __object__ to mention an object to be returned
        # if __import__ is True, then the callable must be imported
        # if not given or False, then we assume the callable is already present
        # in the namespace
        # we check whether the necessary keys are present in the dict keys,
        # by using the issubset method from the set
        # if not satisfied, we raise ValueError
        if not set(self_or_cls.get_keys()).issubset(dict_.keys()):
            # if silent_error is True, then we return the dict without
            # raising the error
            if silent_error:
                return dict_
            else:
                raise ValueError(
                        'The following keys must be '
                        'present to be correctly '
                        'parsed: {}'.format(self_or_cls.get_keys())
                )

        object_str = dict_['__object__']
        import_ = dict_.get('__import__', False)

        object_ = self_or_cls.get_object(
                element=object_str,
                import_=import_
        )

        return object_
