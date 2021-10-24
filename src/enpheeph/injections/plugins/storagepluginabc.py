# -*- coding: utf-8 -*-
import abc
import typing


class StoragePluginABC(abc.ABC):
    @abc.abstractmethod
    def add_element(self, element_name: str, element: typing.Any) -> None:
        pass

    @abc.abstractmethod
    def add_dict(self, dict_: typing.Dict[str, typing.Any]) -> None:
        pass

    # End Of Line, to conclude the current row/list
    @abc.abstractmethod
    def submit_eol(self) -> None:
        pass

    @abc.abstractmethod
    def execute(self) -> None:
        pass
