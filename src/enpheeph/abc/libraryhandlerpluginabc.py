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

import abc
import typing

import enpheeph.injections.abc.injectionabc
import enpheeph.utils.typings


class LibraryHandlerPluginABC(abc.ABC):
    @abc.abstractmethod
    def library_setup(
        self,
        model: enpheeph.utils.typings.ModelType,
        active_injections: typing.List[
            enpheeph.injections.abc.injectionabc.InjectionABC
        ],
    ) -> enpheeph.utils.typings.ModelType:
        pass

    @abc.abstractmethod
    def library_teardown(
        self,
        model: enpheeph.utils.typings.ModelType,
        active_injections: typing.List[
            enpheeph.injections.abc.injectionabc.InjectionABC
        ],
    ) -> enpheeph.utils.typings.ModelType:
        pass
