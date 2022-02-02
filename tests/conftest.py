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

import pytest


# with params we can parametrize the fixture
@pytest.fixture(
    scope="function",
    params=[
        [None, object(), None],
        ["test.module", 2, "test"],
        ["foobar", "a", "foobar"],
        ["second_test", 2, "second_test"],
        [False, [1, 2, 3], None],
    ],
    ids=[
        "None",
        "test.module",
        "foobar",
        "second_test",
        "deletion",
    ],
)
# we need to use request.param to access the parameter
def mock_object_with_library(monkeypatch, request):
    # we get the name of the library to be tested and the object
    library_name, obj, expected_library_name = request.param
    if library_name is not False:
        monkeypatch.setattr(obj.__class__, "__module__", library_name)
    else:
        monkeypatch.delattr(obj.__class__, "__module__")

    return TestWithTarget(test_input=obj, target=expected_library_name)


TestWithTarget = collections.namedtuple("TestWithTarget", "test_input target")
