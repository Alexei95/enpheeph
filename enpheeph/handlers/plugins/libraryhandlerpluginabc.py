import abc
import typing

import enpheeph.injections.injectionabc
import enpheeph.utils.typings


class LibraryHandlerPluginABC(abc.ABC):
    @abc.abstractmethod
    def library_setup(
            self,
            model: enpheeph.utils.typings.ModelType,
            active_injections: typing.List[
                    enpheeph.injections.injectionabc.InjectionABC
            ],
    ) -> enpheeph.utils.typings.ModelType:
        pass

    @abc.abstractmethod
    def library_teardown(
            self,
            model: enpheeph.utils.typings.ModelType,
            active_injections: typing.List[
                    enpheeph.injections.injectionabc.InjectionABC
            ],
    ) -> enpheeph.utils.typings.ModelType:
        pass