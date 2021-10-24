# -*- coding: utf-8 -*-
import abc

import enpheeph.utils.typings


class InjectionABC(abc.ABC):
    @abc.abstractmethod
    def setup(
        self, module: enpheeph.utils.typings.ModelType,
    ) -> enpheeph.utils.typings.ModelType:
        return NotImplemented

    @abc.abstractmethod
    def teardown(
        self, module: enpheeph.utils.typings.ModelType,
    ) -> enpheeph.utils.typings.ModelType:
        return NotImplemented
