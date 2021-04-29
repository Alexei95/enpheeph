import abc
import copy
import functools
import typing


# with this base class we implement a way of dispatching calls to the correct
# callable
# dispatching is actually used for overloaded functions with different
# signatures, but I think this term may be used also in this way
class Dispatcher(object):
    _dispatching_dict: typing.Dict[typing.Any, typing.Callable]

    # the solution was to use a method to return a copy of the dispatching dict
    # while all the operations are done on the hidden class attribute
    @classmethod
    def get_dispatching_dict(cls) -> typing.Dict[typing.Any, typing.Callable]:
        cls._init_dispatching_dict()
        return copy.deepcopy(cls._dispatching_dict)

    # we use this function to init the dispatching dict
    # it must be done in this way so that each subclass has a different dict
    # otherwise it would be the common dict of the ABC class
    @classmethod
    def _init_dispatching_dict(cls):
        # we init the dispatching dict only if it does not exist
        if not hasattr(cls, '_dispatching_dict'):
            cls._dispatching_dict = {}

    # this is the static method to get the name of the callable
    # it is a fallback if the overloading of the virtual method is not done
    @staticmethod
    def callable_name_fallback(callable_: typing.Callable):
        return callable_.__name__

    # this method should be overloaded to customize the dispatcher
    @staticmethod
    def callable_name(callable_: typing.Callable):
        raise NotImplementedError()

    # this decorator registers the decorated callable with the given name
    # or it falls back to the callable name if required
    @classmethod
    def register_decorator(cls, name: typing.Any = None):
        # check and init the dispatching dict
        cls._init_dispatching_dict()

        # we need to add cls and name as arguments for the wrapper
        def wrapper(callable_, *, cls, name):
            # if name is not given it is automated
            if name is None:
                # if the custom callable name is not implemented we fall back
                # to the callable name
                try:
                    name = cls.callable_name(callable_)
                except NotImplementedError:
                    name = cls.callable_name_fallback(callable_)

            # we register the callable
            cls.register(name, callable_)

            # we return the callable as this is a wrapper for the decorator
            return callable_

        # the partial is used so that we can use the wrapper to wrap the
        # decorated function, and we already have cls and name inside the
        # function as arguments
        return functools.partial(wrapper, cls=cls, name=name)

    # with this class method we can register a new callable using a name
    # name is supposed to be either a string or an enum
    @classmethod
    def register(
            cls,
            name: typing.Any,
            callable_: typing.Callable):
        # check and init the dispatching dict
        cls._init_dispatching_dict()

        cls._dispatching_dict[name] = callable_

    # with this class method we can remove a registered callable
    @classmethod
    def deregister(cls, name: typing.Any):
        # check and init the dispatching dict
        cls._init_dispatching_dict()

        del cls._dispatching_dict[name]

    # this is the dispatcher method call, where we dispatch the call to the
    # class to the correct function, with also all the arguments we get extra
    @classmethod
    def dispatch_call(cls, name, *args, **kwargs):
        # check and init the dispatching dict
        cls._init_dispatching_dict()

        return cls._dispatching_dict[name](*args, **kwargs)
