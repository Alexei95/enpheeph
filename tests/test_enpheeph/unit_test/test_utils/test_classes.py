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
import random
import string

import pytest

import enpheeph.utils.classes


class TestClasses(object):
    @pytest.mark.parametrize(
        argnames=("error", "string_param"),
        argvalues=[
            pytest.param(
                ValueError,
                "parameter",
                id="ValueError_parameter",
            ),
            pytest.param(
                TypeError,
                "test",
                id="TypeError_test",
            ),
            pytest.param(
                IndexError,
                None,
                id="IndexError_no_string_to_check",
            ),
            pytest.param(
                (TypeError, ValueError),
                "parameter",
                id="tuple",
            ),
            pytest.param(
                [ValueError, TypeError, OSError],
                "hello",
                id="list",
            ),
        ],
    )
    def test_skip_if_error(self, error, string_param):
        with enpheeph.utils.classes.SkipIfErrorContextManager(
            error=error,
            string_to_check=string_param,
        ):
            a = False
            error = (
                random.choice(error)
                if isinstance(error, collections.abc.Sequence)
                else error
            )
            raise error(
                "".join(
                    random.choice(string.ascii_letters)
                    for _ in range(random.randint(0, 100))
                )
                + string_param
                if string_param is not None
                else random.choice(string.ascii_letters)
                + "".join(
                    random.choice(string.ascii_letters)
                    for _ in range(random.randint(0, 100))
                )
            )
            a = True

        assert not a

    @pytest.mark.parametrize(
        argnames=("error", "string_param"),
        argvalues=[
            pytest.param(
                ValueError,
                "parameter",
                id="ValueError",
            ),
            pytest.param(
                TypeError,
                "test",
                id="TypeError",
            ),
            pytest.param(
                (TypeError, ValueError),
                "parameter",
                id="tuple",
            ),
            pytest.param(
                [TypeError, ValueError],
                "parameter",
                id="list",
            ),
            pytest.param(
                BaseException,
                "parameter",
                id="subclass_should_not_work",
            ),
        ],
    )
    def test_skip_if_error_raising(self, error, string_param):
        with pytest.raises(KeyboardInterrupt):
            with enpheeph.utils.classes.SkipIfErrorContextManager(
                error=error,
                string_to_check=string_param,
            ):
                a = False
                raise KeyboardInterrupt(string_param)
                a = True

        assert not a

    @pytest.mark.parametrize(
        argnames=("error", "string_param"),
        argvalues=[
            pytest.param(
                ["a", TypeError],
                "test",
                id="a_TypeError_test",
            ),
            pytest.param(
                "a",
                "parameter",
                id="a_parameter",
            ),
            pytest.param(
                1,
                "parameter",
                id="1_parameter",
            ),
        ],
    )
    def test_skip_if_error_init_validation_rising(self, error, string_param):
        with pytest.raises(TypeError):
            enpheeph.utils.classes.SkipIfErrorContextManager(
                error=error,
                string_to_check=string_param,
            )

    @pytest.mark.skip(
        reason=(
            "PyTest/unittest do not support mocking __module__ in __class__ "
            "of an object, however this code is left here as memorandum"
        ),
    )
    def test_object_library_with_mocks(self, mock_object_with_library):
        obj, library_name = mock_object_with_library
        assert enpheeph.utils.functions.get_object_library(obj) == library_name

    @pytest.mark.parametrize(
        argnames=("obj", "library_name"),
        argvalues=[
            pytest.param(
                1,
                "builtins",
                id="builtins",
            ),
            pytest.param(
                pytest.hookspec,
                "pluggy",
                id="pluggy_from_pytest",
            ),
            pytest.param(
                collections.defaultdict(),
                "collections",
                id="collections",
            ),
        ],
    )
    def test_object_library_with_real_objs(self, obj, library_name):
        assert enpheeph.utils.functions.get_object_library(obj) == library_name
