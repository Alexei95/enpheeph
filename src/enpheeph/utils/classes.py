# -*- coding: utf-8 -*-
import ast
import inspect
import types
import typing
import weakref


InstanceGeneratorBaseClass = typing.TypeVar(
    "InstanceGeneratorBaseClass", bound="InstanceGeneratorMixin"
)
Subclass = typing.TypeVar('Subclass')


# to define positive numbers
class Positive: ...

# class CustomizableMultiSingletonMetaclass(type, typing.Generic[Subclass]):
#     # we need prepare to receive the keyword arguments during the class creation time
#     @classmethod
#     def __prepare__(metacls, name, bases, **kwargs):
#         singleton_without_init_args = kwargs.get('singleton_without_init_args', False)
#         namespace = {}
#         namespace['singleton_without_init_args'] = singleton_without_init_args

#         return namespace

#     # 
#     def __new__(metacls, name, bases, namespace, **kwargs):
#         singleton_without_init_args = namespace.get(singleton_without_init_args, False)
#         #kwargs = {"myArg1": 1, "myArg2": 2}
#         return super().__new__(metacls, name, bases, namespace, **kwargs)
#         #DO NOT send "**kwargs" to "type.__new__".  It won't catch them and
#         #you'll get a "TypeError: type() takes 1 or 3 arguments" exception.

#     def __init__(cls, name, bases, namespace, **kwargs):
#         super().__init__(name, bases, namespace, **kwargs)

#     _instances = None
#     def __call__(cls, *args: typing.Any, new_instance: bool = False, **kwargs: typing.Any) -> Subclass:
#         if cls._instances is None:
#             cls._instances = weakref.WeakKeyDictionary()
        
#         if cls not in cls._instances:

# # this singleton metaclass is tuned for InjectionLocationABC
# # we need the objects to be hashable, so we can insert them in a weakref dict
# # the key will be the hash of the instance
# # the presence in the set depends on the hash of the object
# # additionally, they will have an id added to them, so that they can be uniquely
# # identified
# class LocationSingletonMetaclass(type, typing.Generic[Subclass]):
#     _instances: typing.Optional[weakref.WeakValueDictionary[int, Subclass]] = None
#     id_counter: int = 0

#     def __call__(cls, *args: typing.Any, new_instance: bool = False, **kwargs: typing.Any) -> Subclass:
#         if cls._instances is None:
#             cls._instances = weakref.WeakValueDictionary()

#         instance = super().__call__(*args, **kwargs)

#         if new_instance or instance not in cls._instances:
#             instance._id = cls.id_counter
#             cls.id_counter += 1
#             cls._instances.add(instance)
        
        
# this metaclass adds an ID to the subclasses
# it also receives an extra argument for forcing a certain ID
# class LocationIDMetaclass(type, typing.Generic[Subclass]):
#     id_counter: int = 0

#     def __call__(cls, *args: typing.Any, force_id: typing.Optional[int] = None, **kwargs: typing.Any) -> Subclass:
#         instance = super().__call__(*args, **kwargs)

#         if force_id is not None:
#             id_ = force_id
#         else:
#             id_ = cls.id_counter
#             cls.id_counter += 1

#         instance.__id = id_

#         return instance


class IDGenerator(object):
    # this is taken from the root if shared
    _ID_COUNTER: typing.Optional[int] = None
    _RESET_VALUE: int = 0
    # this flag is for each class
    _USE_SHARED: bool = False
    # we need a flag to know which one is the root
    # if none of them have it, we resort to the base class, IDGenerator
    _SHARED_ROOT_FLAG: bool = False

    @classmethod
    def __init_subclass__(cls, reset_value: int = _RESET_VALUE, use_shared: bool = _USE_SHARED, shared_root_flag: bool = _SHARED_ROOT_FLAG, **kwargs):
        cls._RESET_VALUE = reset_value
        cls._USE_SHARED = use_shared
        cls._SHARED_ROOT_FLAG = shared_root_flag
        # this call with reset=True is **FUNDAMENTAL** for not sharing the counter
        # otherwise the subclass would receive the setup done on the parent
        # and this would cause the child to have a reference to the parent class 
        # attribute, breaking the independency
        cls._setup_id_counter(reset=True)

        super().__init_subclass__(**kwargs)

    @classmethod
    def _get_root_with_id(cls):
        for cls_ in cls.mro():
            if hasattr(cls_, '_ID_COUNTER') and getattr(cls_, '_SHARED_ROOT_FLAG', False):
                return cls_
        for cls_ in reversed(cls.mro()):
            if hasattr(cls_, '_ID_COUNTER'):
                break
        return cls_

    @classmethod
    def _setup_id_counter(cls, reset: bool = False):
        if cls._USE_SHARED:
            cls_ = cls._get_root_with_id()
        else:
            cls_ = cls
        if reset or cls_._ID_COUNTER is None:
            cls_._ID_COUNTER = cls_._RESET_VALUE

    @classmethod
    def _update_id_counter(cls):
        cls._setup_id_counter(reset=False)
        if cls._USE_SHARED:
            cls_ = cls._get_root_with_id()
        else:
            cls_ = cls
        cls_._ID_COUNTER += 1

    @classmethod
    def _get_id(cls):
        if cls._USE_SHARED:
            cls_ = cls._get_root_with_id()
        else:
            cls_ = cls
        return cls_._ID_COUNTER

    def _set_id(self, reset: bool = False):
        self._setup_id_counter(reset=reset)

        # frozen instance trick
        # no need for the trick in the classmethods
        object.__setattr__(self, '_id', self._get_id())

        self._update_id_counter()

    def __post_init__(self):
        self._set_id()


class FunctionCallerNameMixin(object):
    @classmethod
    def caller_name(cls, depth: typing.Annotated[int, Positive()] = 1) -> str:
        frame: typing.Optional[types.FrameType] = None
        try:
            # we get the name of the called function through the current frame
            # everything is inside a try finally block to delete the frame after the 
            # string has been taken and avoid reference cycles
            frame = inspect.currentframe()
            if frame is None:
                raise RuntimeError("Must be run on CPython, as other implementations are not currently supported")
            for i in range(depth + 1):
                frame_new = frame.f_back
                del frame
                frame = frame_new
                if frame is None:
                    raise RuntimeError("Must be run on CPython, as other implementations are not currently supported")
            # we need to get the calling frame f_back for the current one before getting
            # the function name co_name from its code f_code
            name: str = frame.f_code.co_name
        finally:
            del frame

        return name


class InstanceGeneratorMixin(object):
    # this method does not work with enums or with nested function calls
    @classmethod
    def from_safe_repr(
        cls: typing.Type[InstanceGeneratorBaseClass], representation: str
    ) -> InstanceGeneratorBaseClass:
        # we assume only one call
        call_element = ast.parse(representation).body[0].value
        # we assume function  name is identical to the cls.__qualname__
        assert cls.__qualname__ == call_element.func.id
        # we parse arguments and keyword-arguments using ast.literal_eval
        args = []
        for arg in call_element.args:
            args.append(ast.literal_eval(arg))
        kwargs = {}
        for keyword_element in call_element.keywords:
            kwargs[keyword_element.arg] = ast.literal_eval(keyword_element.value,)
        return cls(*args, **kwargs)
