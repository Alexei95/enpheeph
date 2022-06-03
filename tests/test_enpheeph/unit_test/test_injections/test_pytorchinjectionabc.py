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

import enpheeph.injections.pytorchinjectionabc


class TestPyTorchInjectionABC(object):
    def test_abstract_method_setup(self):
        assert getattr(
            enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC.setup,
            "__isabstractmethod__",
            False,
        )

    def test_teardown(self):
        # NOTE: teardown cannot be tested directly as the class cannot be instantiated
        # directly due to the abstract methods
        assert not getattr(
            enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC.teardown,
            "__isabstractmethod__",
            False,
        )

    def test_abstract_module_name(self):
        assert getattr(
            enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC.module_name,
            "__isabstractmethod__",
            False,
        )

        # we check whether the method is a property
        assert isinstance(
            enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC.module_name,
            property,
        )
