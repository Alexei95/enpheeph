# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2023 Alessio "Alexei95" Colucci
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

import enpheeph.utils.constants
import enpheeph.utils.dataclasses
import enpheeph.utils.enums
import enpheeph.utils.typings


class IndexingPluginABC(abc.ABC):
    active_dimension_index: typing.Optional[
        typing.List[enpheeph.utils.typings.ActiveDimensionIndexType]
    ]
    dimension_dict: enpheeph.utils.typings.DimensionDictType

    # to select a set of dimensions to be used as active when selecting tensor indices
    # by default no dimension is considered active
    @abc.abstractmethod
    def select_active_dimensions(
        self,
        dimensions: typing.Sequence[enpheeph.utils.enums.DimensionType],
        # if True, we will move all the indices so that the first index is 0
        # and the last is -1
        autoshift_to_boundaries: bool = False,
        # if True we fill the empty indices with the filler
        # if False we will skip them
        fill_empty_index: bool = True,
        # the filler to use, defaults to : for a single dimension,
        # which is slice(None, None)
        filler: typing.Any = slice(None, None),
    ) -> typing.List[enpheeph.utils.typings.ActiveDimensionIndexType]:
        pass

    # to reset the active dimensions to the empty dimension dict
    @abc.abstractmethod
    def reset_active_dimensions(self) -> None:
        pass

    # to join indices following the order provided by the active_dimension dict
    @abc.abstractmethod
    def join_indices(
        self,
        dimension_indices: enpheeph.utils.typings.DimensionLocationIndexType,
    ) -> enpheeph.utils.typings.AnyIndexType:
        pass

    # to filter a size/shape array depending on the active dimension index
    # by selecting only the dimensions with the enum
    @abc.abstractmethod
    def filter_dimensions(
        self,
        # a normal size/shape array
        dimensions: typing.Sequence[int],
    ) -> typing.Tuple[int, ...]:
        pass
