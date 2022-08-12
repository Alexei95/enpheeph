# -*- coding: utf-8 -*-

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

import collections
import typing


# it overwrites the keys with the new value
# in case of list, str, dict, it tries to merge them together
def recursive_dict_update(original: typing.Dict, mergee: typing.Dict) -> typing.Dict:
    for k, v in mergee.items():
        if k in original and isinstance(original[k], collections.abc.Mapping):
            original[k] = recursive_dict_update(original[k], v)
        else:
            original[k] = v
    return original
