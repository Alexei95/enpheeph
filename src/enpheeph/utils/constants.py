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

import enpheeph.utils.enums
import enpheeph.utils.typings


NORSE_DIMENSION_DICT: enpheeph.utils.typings.DimensionDictType = {
    enpheeph.utils.enums.DimensionType.Time: 0,
    enpheeph.utils.enums.DimensionType.Batch: 1,
    enpheeph.utils.enums.DimensionType.Tensor: ...,
}
PYTORCH_DIMENSION_DICT: enpheeph.utils.typings.DimensionDictType = {
    enpheeph.utils.enums.DimensionType.Batch: 0,
    enpheeph.utils.enums.DimensionType.Tensor: ...,
}
