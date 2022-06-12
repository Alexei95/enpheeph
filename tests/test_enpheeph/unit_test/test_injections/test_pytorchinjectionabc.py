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

import torch

import enpheeph.injections.pytorchinjectionabc


class TestPyTorchInjectionABC(object):
    def test_abstract_method_setup(self):
        assert getattr(
            enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC.setup,
            "__isabstractmethod__",
            False,
        )

    def test_teardown_not_abstract(self):
        assert not getattr(
            enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC.teardown,
            "__isabstractmethod__",
            False,
        )

    def test_teardown(self):
        class Implementation(
            enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC
        ):
            def setup(self):
                pass

            def module_name(self):
                pass

        instance = Implementation()
        module = torch.nn.ReLU()

        module = instance.teardown(module)

        assert module(torch.tensor([1])) == torch.tensor([1])

        instance.handle = module.register_forward_hook(lambda m, i, o: o + 1)

        assert module(torch.tensor([1])) == torch.tensor([2])

        module = instance.teardown(module)

        assert module(torch.tensor([1])) == torch.tensor([1])

        module = instance.teardown(module)

        assert module(torch.tensor([1])) == torch.tensor([1])

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

    def test_attributes(self):
        class_ = enpheeph.injections.pytorchinjectionabc.PyTorchInjectionABC
        # __annotations__ returns the annotated attributes in the class
        assert "handle" in class_.__annotations__
