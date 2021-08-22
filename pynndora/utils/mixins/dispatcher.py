import abc
import copy
import functools
import inspect
import typing

import src.utils.instance_or_classmethod


# with this base class we implement a way of dispatching calls to the correct
# callable
# dispatching is actually used for overloaded functions with different
# signatures, but I think this term may be used also in this way
class Dispatcher(object):
    _dispatching_dict: typing.Dict[typing.Any, typing.Callable]

    # the solution was to use a method to return a copy of the dispatching dict
    # while all the operations are done on the hidden class attribute
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def get_dispatching_dict(self_of_cls) -> typing.Dict[
            typing.Any, typing.Callable
    ]:
        self_of_cls._init_dispatching_dict()
        return copy.deepcopy(self_of_cls._dispatching_dict)

    # we use this function to init the dispatching dict
    # it must be done in this way so that each subclass has a different dict
    # otherwise it would be the common dict of the ABC class
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def _init_dispatching_dict(self_of_cls):
        # we init the dispatching dict only if it does not exist
        if not hasattr(self_of_cls, '_dispatching_dict'):
            self_of_cls._dispatching_dict = {}

    # this is the static method to get the name of the callable
    # it is a fallback if the overloading of the virtual method is not done
    # use qualname instead of name for classes, to allow nested classes
    @staticmethod
    def callable_name_fallback(callable_: typing.Callable):
        return callable_.__qualname__

    # this method should be overloaded to customize the dispatcher
    @staticmethod
    def callable_name(callable_: typing.Callable):
        raise NotImplementedError()

    # this decorator registers the decorated callable with the given name
    # or it falls back to the callable name if required
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def register_decorator(self_of_cls, name: typing.Any = None):
        # check and init the dispatching dict
        self_of_cls._init_dispatching_dict()

        # we need to add cls/self and name as arguments for the wrapper
        def wrapper(callable_, *, self_of_cls, name):
            # if name is not given it is automated
            if name is None:
                # if the custom callable name is not implemented we fall back
                # to the callable name
                try:
                    name = self_of_cls.callable_name(callable_)
                except NotImplementedError:
                    name = self_of_cls.callable_name_fallback(callable_)

            # we register the callable
            self_of_cls.register(name, callable_)

            # we return the callable as this is a wrapper for the decorator
            return callable_

        # the partial is used so that we can use the wrapper to wrap the
        # decorated function, and we already have cls and name inside the
        # function as arguments
        return functools.partial(wrapper, self_of_cls=self_of_cls, name=name)

    # with this class method we can register a new callable using a name
    # name is supposed to be either a string or an enum
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def register(
            self_of_cls,
            name: typing.Any,
            callable_: typing.Callable):
        # check and init the dispatching dict
        self_of_cls._init_dispatching_dict()

        self_of_cls._dispatching_dict[name] = callable_

    # with this class method we can remove a registered callable
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def deregister(self_of_cls, name: typing.Any):
        # check and init the dispatching dict
        self_of_cls._init_dispatching_dict()

        del self_of_cls._dispatching_dict[name]

    # this is the dispatcher method call, where we dispatch the call to the
    # class to the correct function, with also all the arguments we get extra
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def dispatch_call(self_of_cls, name, *args, **kwargs):
        # check and init the dispatching dict
        self_of_cls._init_dispatching_dict()

        return self_of_cls._dispatching_dict[name](*args, **kwargs)

    # this method is used to register similar method names, based around
    # the same template
    @src.utils.instance_or_classmethod.instance_or_classmethod
    def register_string_methods(
            self_of_cls,
            object_: typing.Any,
            template_string: str,
            string_list: typing.Sequence[str],
            name_list: typing.Sequence[str],
            # if the getattr returns None used as default value
            # we raise an error if this flag is True
            error_if_none: bool = False,
    ):
        # we assume the two lists contain the same number of elements
        for name, string in zip(name_list, string_list):
            method_string = template_string.format(string)
            method = getattr(object_, method_string, None)
            if method is None and error_if_none:
                raise ValueError(
                        f"'{method_string}' "
                        "does not exist in the given object"
                )
            self_of_cls.register(name, method)
