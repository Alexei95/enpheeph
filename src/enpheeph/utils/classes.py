# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2026 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import collections.abc
import types
import typing


IDGeneratorSubclass = typing.TypeVar("IDGeneratorSubclass", bound="IDGenerator")


# base class for generating sequential IDs for different instances
# there is the possibility of setting the start value, as well as the sharing of the
# different flags with a common base class
class IDGenerator(object):
    # these are the defaults for all the options
    # this is taken from the root if shared
    _INSTANCE_ID_COUNTER: typing.Optional[int] = None
    _INSTANCE_ID_COUNTER_RESET_VALUE: int = 0
    # this flag is for each class
    _INSTANCE_ID_COUNTER_USE_SHARED: bool = False
    # we need a flag to know which one is the root
    # if none of them have it, we resort to the base class, IDGenerator
    _INSTANCE_ID_COUNTER_SHARED_ROOT_FLAG: bool = False

    # we define the typing for each class instance, to avoid mypy errors
    _unique_instance_id: int

    @property
    def unique_instance_id(self) -> int:
        return self._unique_instance_id

    # we override init_subclass, to get the arguments from the class definition
    # we can set the reset value for the counter (starting value)
    # we can also set on a per-class basis whether the class is to be considered a
    # root and whether it should use a shared counter or use its own
    @classmethod
    def __init_subclass__(
        cls: typing.Type[IDGeneratorSubclass],
        reset_value: int = _INSTANCE_ID_COUNTER_RESET_VALUE,
        use_shared: bool = _INSTANCE_ID_COUNTER_USE_SHARED,
        shared_root_flag: bool = _INSTANCE_ID_COUNTER_SHARED_ROOT_FLAG,
        **kwargs: typing.Any,
    ) -> None:
        # we set the class defaults overriding the root defaults
        cls._INSTANCE_ID_COUNTER_RESET_VALUE = reset_value
        cls._INSTANCE_ID_COUNTER_USE_SHARED = use_shared
        cls._INSTANCE_ID_COUNTER_SHARED_ROOT_FLAG = shared_root_flag
        # this call with reset=True is **FUNDAMENTAL** for not sharing the counter
        # otherwise the subclass would receive the setup done on the parent
        # and this would cause the child to have a reference to the parent class
        # attribute, breaking the independency
        cls._setup_id_counter(reset=True)

        # we ignore the problem with object.__init_subclass__
        # this class is supposed to be sub-classed, so it will handle general kwargs
        # for other parent classes
        super().__init_subclass__(**kwargs)  # type: ignore[call-arg]

    # we have to use the shared flag if the flag is set
    # we go through the mros (which are from the most specific class backward to object)
    # to reach a class which has the root flag enabled
    # if this does not happen, we go through the mros from object down until we find
    # the deepest root which has an id counter
    @classmethod
    def _get_root_with_id(
        cls: typing.Type[IDGeneratorSubclass],
    ) -> typing.Type[IDGeneratorSubclass]:
        if cls._INSTANCE_ID_COUNTER_USE_SHARED:
            for cls_ in cls.mro():
                if hasattr(cls_, "_INSTANCE_ID_COUNTER") and getattr(
                    cls_, "_INSTANCE_ID_COUNTER_SHARED_ROOT_FLAG", False
                ):
                    return cls_
            for cls_ in reversed(cls.mro()):
                if hasattr(cls_, "_INSTANCE_ID_COUNTER"):
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
        if reset or cls_._INSTANCE_ID_COUNTER is None:
            cls_._INSTANCE_ID_COUNTER = cls_._INSTANCE_ID_COUNTER_RESET_VALUE

    # we update the counter in the correct class
    @classmethod
    def _update_id_counter(cls: typing.Type[IDGeneratorSubclass]) -> None:
        cls_ = cls._get_root_with_id()
        cls_._setup_id_counter(reset=False)
        # we ignore this type error as we setup the id counter in the previous line
        cls_._INSTANCE_ID_COUNTER += 1  # type: ignore[operator]

    # to return the id counter
    @classmethod
    def _get_id_counter(cls: typing.Type[IDGeneratorSubclass]) -> typing.Optional[int]:
        cls_ = cls._get_root_with_id()
        return cls_._INSTANCE_ID_COUNTER

    # to set the id in the current instance
    # this is supposed to be called during __new__, to set the instance id after the
    # reset
    def _set_instance_id(self: IDGeneratorSubclass, reset: bool = False) -> None:
        self._setup_id_counter(reset=reset)

        # frozen instance trick
        # no need for the trick in the classmethods
        object.__setattr__(self, "_unique_instance_id", self._get_id_counter())

        self._update_id_counter()

    # we override new to set the instance id
    def __new__(
        cls: typing.Type[IDGeneratorSubclass], *args: typing.Any, **kwargs: typing.Any
    ) -> IDGeneratorSubclass:
        obj: IDGeneratorSubclass = super().__new__(cls)
        obj._set_instance_id()
        return obj


class SkipIfErrorContextManager(object):
    def __init__(
        self,
        # use typing.Type as type is not subscriptable in Python 3.8
        error: typing.Union[
            typing.Type[BaseException], typing.Sequence[typing.Type[BaseException]]
        ],
        string_to_check: typing.Optional[str] = None,
    ) -> None:
        # we save the error in a tuple if it is a single class
        if not isinstance(error, collections.abc.Sequence):
            error = (error,)
        error = tuple(error)
        # we check for each element to be a BaseException subclass
        for e in error:
            if not issubclass(e, BaseException):
                raise TypeError(f"Not a valid BaseException subclass: {e}")

        self.error = error
        self.string_to_check = string_to_check

    def __enter__(self) -> None:
        pass

    # how to type a context manager
    # https://adamj.eu/tech/2021/07/04/python-type-hints-how-to-type-a-context-manager/
    def __exit__(
        self,
        # use typing.Type as type is not subscriptable in Python 3.8
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_val: typing.Optional[BaseException],
        exc_tb: typing.Optional[types.TracebackType],
    ) -> typing.Optional[bool]:
        # if we have received the error to be caught with its string, we return True
        # to avoid the error from propagating
        if exc_type is not None and exc_val is not None:
            error_presence = exc_type in self.error
            string_check = (
                self.string_to_check in str(exc_val)
                if self.string_to_check is not None
                else True
            )
            return error_presence and string_check
