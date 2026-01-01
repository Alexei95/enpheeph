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
import copy
import typing

import enpheeph.injections.plugins.indexing.abc.indexingpluginabc
import enpheeph.utils.constants
import enpheeph.utils.dataclasses
import enpheeph.utils.enums
import enpheeph.utils.typings


class IndexingPlugin(
    enpheeph.injections.plugins.indexing.abc.indexingpluginabc.IndexingPluginABC
):
    # it is Optional so that we can use None
    active_dimension_index: typing.Optional[
        typing.List[enpheeph.utils.typings.ActiveDimensionIndexType]
    ]
    dimension_dict: enpheeph.utils.typings.DimensionDictType

    def __init__(
        self, dimension_dict: enpheeph.utils.typings.DimensionDictType
    ) -> None:
        self.dimension_dict = dimension_dict

        self.reset_active_dimensions()

    # to select a set of dimensions to be used as active when selecting tensor indices
    # by default no dimension is considered active
    def select_active_dimensions(
        self,
        dimensions: collections.abc.Container[enpheeph.utils.enums.DimensionType],
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
        # we invert the dimension dict to easily look it up
        # as we will be using the indices to look it up instead of the names
        inverted_dimension_dict = {v: k for k, v in self.dimension_dict.items()}
        # we get the highest index for both the positive and the negative indices
        # in terms of absolute value
        # we filter the Ellipsis to avoid mypy errors
        # **NOTE**: improve the typing here
        no_ellipsis_dimension_dict_values: typing.List[int] = typing.cast(
            typing.List[int,],
            [x for x in self.dimension_dict.values() if x != Ellipsis],
        )
        longest_positive_range: int = max(
            (x for x in no_ellipsis_dimension_dict_values if x >= 0),
            # we use -1 default so that range(-1 + 1) = []
            default=-1,
        )
        longest_negative_range: int = min(
            (x for x in no_ellipsis_dimension_dict_values if x < 0),
            # we use the number right outside the range to get an empty list
            default=0,
        )
        # this list contains all the possible indices including Ellipsis
        total_indices: typing.List[enpheeph.utils.typings.DimensionIndexType] = list(
            # we cover all the indices to the maximum,
            # including the maximum itself,
            # hence the + 1
            range(longest_positive_range + 1),
        )
        # we need to split the list creation otherwise mypy complains of different types
        total_indices += [Ellipsis]
        total_indices += list(
            # we create the list going from the most negative index to 0
            # 0 is excluded
            range(
                longest_negative_range,
                0,
            ),
        )
        # we save the filling and the valid indices in the following list
        dimension_index: typing.List[
            enpheeph.utils.typings.ActiveDimensionIndexType,
        ] = []
        for index in total_indices:
            # the index is saved if it is present in the dimensions to be selected
            # here we still don't consider the autoshift
            if (
                index in inverted_dimension_dict
                and inverted_dimension_dict[index] in dimensions
            ):
                dimension_index.append(inverted_dimension_dict[index])
            # if the index is not included, we then check if we need to fill it
            # due to fill_empty_index
            elif fill_empty_index:
                dimension_index.append(filler)
        if autoshift_to_boundaries:
            # we remove all the elements at the beginning/end of the list
            # that are fillers
            i = 0
            # infinite loop, but there is a break
            # **NOTE**: probably it can be optimized further
            while 1:
                # we start from 0, and for each filler we match we remove it
                if dimension_index[i] == filler:
                    del dimension_index[i]
                # if the element is not a filler than the start is done and we check the
                # end using -1
                elif i == 0:
                    i = -1
                # if both the element is not a filler and the index is at the end, it
                # means we are done
                else:
                    break
        # we copy the dimensions and we return them
        self.active_dimension_index = copy.deepcopy(dimension_index)
        return copy.deepcopy(self.active_dimension_index)

    # to reset the active dimensions to the empty dimension dict
    def reset_active_dimensions(self) -> None:
        self.active_dimension_index = None

    # to join indices following the order provided by the active_dimension dict
    def join_indices(
        self,
        dimension_indices: enpheeph.utils.typings.DimensionLocationIndexType,
    ) -> enpheeph.utils.typings.AnyIndexType:
        if self.active_dimension_index is None:
            raise ValueError(
                "First select the active dimensions with select_active_dimensions"
            )

        index: typing.List[enpheeph.utils.typings.Index1DType] = []
        for i in self.active_dimension_index:
            # if we have an enum as index we check it from the given dimensions
            if isinstance(i, enpheeph.utils.enums.DimensionType):
                # to check if we have a sequence of sequence we want each element
                # to be a sequence and have no elements which are integers, as
                # the other allowed values represent sequences
                sequence_of_sequence = isinstance(
                    dimension_indices[i], collections.abc.Sequence
                ) and not any(
                    isinstance(j, int)
                    # we use typing.cast to avoid mypy complaining
                    for j in typing.cast(
                        typing.Sequence[typing.Any],
                        dimension_indices[i],
                    )
                )
                # if it is a sequence of sequences we extend the index with all the
                # sub-sequences, as it will cover multiple dimensions
                if sequence_of_sequence:
                    index.extend(
                        typing.cast(
                            typing.Tuple[enpheeph.utils.typings.Index1DType, ...],
                            dimension_indices[i],
                        ),
                    )
                # otherwise it covers only 1 dimension so we append the element directly
                else:
                    index.append(
                        typing.cast(
                            enpheeph.utils.typings.Index1DType,
                            dimension_indices[i],
                        ),
                    )
            # if the element is not an enum it will be a filler,
            # so we append it directly
            else:
                index.append(i)
        return copy.deepcopy(tuple(index))

    # to filter a size/shape array depending on the active dimension index
    # by selecting only the dimensions with the enum
    def filter_dimensions(
        self,
        # a normal size/shape array
        dimensions: typing.Sequence[int],
    ) -> typing.Tuple[int, ...]:
        if self.active_dimension_index is None:
            raise ValueError(
                "First select the active dimensions with select_active_dimensions"
            )

        enum_types = [
            e
            for e in self.active_dimension_index
            if isinstance(e, enpheeph.utils.enums.DimensionType)
        ]
        active_dimension_index: typing.List[
            enpheeph.utils.typings.ActiveDimensionIndexType
        ] = copy.deepcopy(self.active_dimension_index)
        for e in enum_types:
            if self.dimension_dict[e] == Ellipsis:
                while len(dimensions) > len(active_dimension_index):
                    active_dimension_index.insert(active_dimension_index.index(e), e)
        # this is executed if the loop exits normally
        else:
            if len(dimensions) != len(active_dimension_index):
                raise ValueError(
                    "dimensions must be the same length of active_dimension_index "
                    "if no Ellipsis are used"
                )

        return_dimensions = []
        for d, ind in zip(dimensions, active_dimension_index):
            if isinstance(ind, enpheeph.utils.enums.DimensionType):
                return_dimensions.append(d)
        return tuple(return_dimensions)
