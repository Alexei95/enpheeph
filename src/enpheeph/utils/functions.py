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

import re
import typing


CAMEL_TO_SNAKE_REGEX: re.Pattern[str] = re.compile(
    "((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))"
)


# this function is required to convert CamelCase to snake_case
def camel_to_snake(camel: str) -> str:
    # from https://stackoverflow.com/a/12867228
    return CAMEL_TO_SNAKE_REGEX.sub(r"_\1", camel).lower()


def get_object_library(obj: typing.Any) -> str | None:
    module = getattr(obj.__class__, "__module__", None)
    # to be safe we return None if the module is not a string
    return module.split(".")[0] if isinstance(module, str) else None
