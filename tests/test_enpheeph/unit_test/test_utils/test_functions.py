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

import collections

import pytest
import pytest_cases

import enpheeph.utils.functions


class CasesCamelCaseToSnakeCaseFunction(object):
    def case_CamelSnake(self):
        return ("CamelSnake", "camel_snake")

    def case_camelSnake(self):
        return ("camelSnake", "camel_snake")

    def case_camel_snake(self):
        return ("camel_snake", "camel_snake")


class CasesGetObjectLibraryFunction(object):
    def case_int_from_builtins(self):
        return (1, "builtins")

    def case_pluggy_from_pytest(self):
        return (pytest.hookspec, "pluggy")

    def case_defaultdict_from_collections(self):
        return (collections.defaultdict(), "collections")


class TestCamelCaseToSnakeCaseFunction(object):
    @pytest_cases.parametrize_with_cases(
        argnames=("camel", "snake"),
        cases=CasesCamelCaseToSnakeCaseFunction,
    )
    def test_camel_to_snake(self, camel, snake):
        assert enpheeph.utils.functions.camel_to_snake(camel) == snake


class TestGetObjectLibraryFunction(object):
    @pytest.mark.skip(
        reason=(
            "PyTest/unittest do not support mocking __module__ in __class__ "
            "of an object, however this code is left here as memorandum"
        ),
    )
    def test_get_object_library_with_mocks(self, mock_object_with_library):
        obj, library_name = mock_object_with_library
        assert enpheeph.utils.functions.get_object_library(obj) == library_name

    @pytest_cases.parametrize_with_cases(
        argnames="obj,library_name",
        cases=CasesGetObjectLibraryFunction,
    )
    def test_get_object_library_with_real_objs(self, obj, library_name):
        assert enpheeph.utils.functions.get_object_library(obj) == library_name
