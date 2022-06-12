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


import enpheeph.injections.injectionabc


class TestInjectionABC(object):
    def test_abstract_method_setup(self):
        assert getattr(
            enpheeph.injections.injectionabc.InjectionABC.setup,
            "__isabstractmethod__",
            False,
        )

    def test_abstract_method_teardown(self):
        assert getattr(
            enpheeph.injections.injectionabc.InjectionABC.teardown,
            "__isabstractmethod__",
            False,
        )

    def test_attributes(self):
        # __annotations__ returns the annotated attributes in the class
        assert (
            "location" in enpheeph.injections.injectionabc.InjectionABC.__annotations__
        )
