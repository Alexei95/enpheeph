# -*- coding: utf-8 -*-
import inspect
import types
import typing


IDGeneratorSubclass = typing.TypeVar("IDGeneratorSubclass", bound="IDGenerator")


# to define positive numbers
class Positive:
    pass


# base class for generating sequential IDs for different instances
# there is the possibility of setting the start value, as well as the sharing of the
# different flags with a common base class
class IDGenerator(object):
    # these are the defaults for all the options
    # this is taken from the root if shared
    _ID_COUNTER: typing.Optional[int] = None
    _RESET_VALUE: int = 0
    # this flag is for each class
    _USE_SHARED: bool = False
    # we need a flag to know which one is the root
    # if none of them have it, we resort to the base class, IDGenerator
    _SHARED_ROOT_FLAG: bool = False

    # we override init_subclass, to get the arguments from the class definition
    # we can set the reset value for the counter (starting value)
    # we can also set on a per-class basis whether the class is to be considered a
    # root and whether it should use a shared counter or use its own
    @classmethod
    def __init_subclass__(
        cls: typing.Type[IDGeneratorSubclass],
        reset_value: int = _RESET_VALUE,
        use_shared: bool = _USE_SHARED,
        shared_root_flag: bool = _SHARED_ROOT_FLAG,
        **kwargs: typing.Any,
    ) -> None:
        # we set the class defaults overriding the root defaults
        cls._RESET_VALUE = reset_value
        cls._USE_SHARED = use_shared
        cls._SHARED_ROOT_FLAG = shared_root_flag
        # this call with reset=True is **FUNDAMENTAL** for not sharing the counter
        # otherwise the subclass would receive the setup done on the parent
        # and this would cause the child to have a reference to the parent class
        # attribute, breaking the independency
        cls._setup_id_counter(reset=True)

        # we ignore the problem with object.__init_subclass__
        # this class is supposed to be sub-classed, so it will handle general kwargs
        # for other parent classes
        super().__init_subclass__(**kwargs)  # type: ignore

    # we have to use the shared flag if the flag is set
    # we go through the mros (which are from lowest to object) to reach a class which
    # has the root flag enabled
    # if this does not happen, we go through the mros from object down until we find
    # the deepest root which has an id counter
    @classmethod
    def _get_root_with_id(
        cls: typing.Type[IDGeneratorSubclass],
    ) -> typing.Type[IDGeneratorSubclass]:
        if cls._USE_SHARED:
            for cls_ in cls.mro():
                if hasattr(cls_, "_ID_COUNTER") and getattr(
                    cls_, "_SHARED_ROOT_FLAG", False
                ):
                    return cls_
            for cls_ in reversed(cls.mro()):
                if hasattr(cls_, "_ID_COUNTER"):
                    return cls_
        return cls

    # we setup the counter, which is reset to the original value if the counter is
    # initially None or it is forced
    # **IT IS FUNDAMENTAL** to run it with reset so that each class counter is set
    # otherwise it will use the root one in a shared configuration
    @classmethod
    def _setup_id_counter(
        cls: typing.Type[IDGeneratorSubclass], reset: bool = False
    ) -> None:
        cls_ = cls._get_root_with_id()
        if reset or cls_._ID_COUNTER is None:
            cls_._ID_COUNTER = cls_._RESET_VALUE

    # we update the counter in the correct class
    @classmethod
    def _update_id_counter(cls: typing.Type[IDGeneratorSubclass]) -> None:
        cls_ = cls._get_root_with_id()
        cls_._setup_id_counter(reset=False)
        # we ignore this type error as we setup the id counter in the previous line
        cls_._ID_COUNTER += 1  # type: ignore

    # to return the id counter
    @classmethod
    def _get_id_counter(cls: typing.Type[IDGeneratorSubclass]) -> typing.Optional[int]:
        cls_ = cls._get_root_with_id()
        return cls_._ID_COUNTER

    # to set the id in the current instance
    # this is supposed to be called during __new__, to set the instance id after the
    # reset
    def _set_instance_id(self: IDGeneratorSubclass, reset: bool = False) -> None:
        self._setup_id_counter(reset=reset)

        # frozen instance trick
        # no need for the trick in the classmethods
        object.__setattr__(self, "_id", self._get_id_counter())

        self._update_id_counter()

    # we override new to set the instance id
    def __new__(
        cls: typing.Type[IDGeneratorSubclass], *args: typing.Any, **kwargs: typing.Any
    ) -> IDGeneratorSubclass:
        obj: IDGeneratorSubclass = super().__new__(cls)
        obj._set_instance_id()
        return obj


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
                raise RuntimeError(
                    "Must be run on CPython, as other implementations "
                    "are not currently supported"
                )
            for i in range(depth + 1):
                frame_new = frame.f_back
                del frame
                frame = frame_new
                if frame is None:
                    raise RuntimeError(
                        "Must be run on CPython, as other implementations "
                        "are not currently supported"
                    )
            # we need to get the calling frame f_back for the current one before getting
            # the function name co_name from its code f_code
            name: str = frame.f_code.co_name
        finally:
            del frame

        return name
