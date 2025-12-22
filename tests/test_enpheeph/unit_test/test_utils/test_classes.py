# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2025 Alessio "Alexei95" Colucci
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
import random
import string

import pytest

import enpheeph.utils.classes


class TestIDGeneratorClass(object):
    def test_increasing_id_for_same_class_objects(self):
        class A(
            enpheeph.utils.classes.IDGenerator,
            use_shared=False,
            reset_value=0,
            shared_root_flag=True,
        ):
            pass

        a1 = A()
        a2 = A()

        assert a1.unique_instance_id == 0
        assert a2.unique_instance_id == 1

    def test_increasing_id_for_shared_subclasses(self):
        class A(
            enpheeph.utils.classes.IDGenerator,
            use_shared=True,
            reset_value=0,
            shared_root_flag=True,
        ):
            pass

        class B(A, use_shared=True, reset_value=0, shared_root_flag=False):
            pass

        a1 = A()
        a2 = A()
        b1 = B()
        b2 = B()
        a3 = A()
        b3 = B()

        assert a1.unique_instance_id == 0
        assert a2.unique_instance_id == a1.unique_instance_id + 1
        assert b1.unique_instance_id == a2.unique_instance_id + 1
        assert b2.unique_instance_id == b1.unique_instance_id + 1
        assert a3.unique_instance_id == b2.unique_instance_id + 1
        assert b3.unique_instance_id == a3.unique_instance_id + 1

    def test_increasing_independent_id_for_shared_subclasses(self):
        class A(
            enpheeph.utils.classes.IDGenerator,
            use_shared=True,
            reset_value=0,
            shared_root_flag=True,
        ):
            pass

        class B(A, use_shared=False, reset_value=0, shared_root_flag=False):
            pass

        a1 = A()
        a2 = A()
        b1 = B()
        b2 = B()
        a3 = A()
        b3 = B()

        assert a1.unique_instance_id == 0
        assert a2.unique_instance_id == 1
        assert b1.unique_instance_id == 0
        assert b2.unique_instance_id == 1
        assert a3.unique_instance_id == 2
        assert b3.unique_instance_id == 2

    def test_different_reset_value_id(self):
        class A(
            enpheeph.utils.classes.IDGenerator,
            use_shared=True,
            reset_value=10,
            shared_root_flag=True,
        ):
            pass

        class B(A, use_shared=False, reset_value=23, shared_root_flag=False):
            pass

        a1 = A()
        a2 = A()
        b1 = B()
        b2 = B()
        a3 = A()
        b3 = B()

        assert a1.unique_instance_id == 10
        assert a2.unique_instance_id == 11
        assert b1.unique_instance_id == 23
        assert b2.unique_instance_id == 24
        assert a3.unique_instance_id == 12
        assert b3.unique_instance_id == 25

    def test_get_root_with_counter(self):
        class A(
            enpheeph.utils.classes.IDGenerator, use_shared=False, shared_root_flag=True
        ):
            pass

        class B(
            enpheeph.utils.classes.IDGenerator, use_shared=True, shared_root_flag=False
        ):
            pass

        class C(A, use_shared=False, shared_root_flag=False):
            pass

        class D(C, use_shared=True, shared_root_flag=False):
            pass

        class E(
            enpheeph.utils.classes.IDGenerator, use_shared=False, shared_root_flag=False
        ):
            pass

        class F(
            enpheeph.utils.classes.IDGenerator, use_shared=True, shared_root_flag=False
        ):
            pass

        class G(
            enpheeph.utils.classes.IDGenerator, use_shared=True, shared_root_flag=True
        ):
            pass

        assert A._get_root_with_id() == A
        assert B._get_root_with_id() == enpheeph.utils.classes.IDGenerator
        assert C._get_root_with_id() == C
        assert D._get_root_with_id() == A
        assert E._get_root_with_id() == E
        assert F._get_root_with_id() == enpheeph.utils.classes.IDGenerator
        assert G._get_root_with_id() == G

    def test_setup_id_counter(self):
        class A(
            enpheeph.utils.classes.IDGenerator,
            reset_value=10,
            use_shared=False,
            shared_root_flag=True,
        ):
            pass

        class B(A, use_shared=True, shared_root_flag=False):
            pass

        class C(A, use_shared=False, shared_root_flag=False):
            pass

        A._setup_id_counter(reset=True)
        assert A().unique_instance_id == 10
        assert B().unique_instance_id == 11
        A._setup_id_counter(reset=False)
        B._setup_id_counter(reset=False)
        assert A().unique_instance_id == 12
        assert B().unique_instance_id == 13
        B._setup_id_counter(reset=True)
        assert B().unique_instance_id == 10
        assert A().unique_instance_id == 11

        C._setup_id_counter(reset=True)
        assert C().unique_instance_id == 0
        C._setup_id_counter(reset=False)
        assert C().unique_instance_id == 1

    def test_update_id_counter(self):
        class A(
            enpheeph.utils.classes.IDGenerator,
            reset_value=10,
            use_shared=False,
            shared_root_flag=True,
        ):
            pass

        class B(A, use_shared=True, shared_root_flag=False):
            pass

        class C(A, use_shared=False, shared_root_flag=False):
            pass

        assert A().unique_instance_id == 10
        A._setup_id_counter(reset=True)
        A._update_id_counter()
        assert A().unique_instance_id == 11
        A._update_id_counter()
        A._update_id_counter()
        assert A().unique_instance_id == 14

        assert B().unique_instance_id == 15
        B._update_id_counter()
        assert A().unique_instance_id == 17
        assert B().unique_instance_id == 18

        assert C().unique_instance_id == 0
        C._update_id_counter()
        assert C().unique_instance_id == 2

    def test_get_id_counter(self):
        class A(
            enpheeph.utils.classes.IDGenerator,
            reset_value=10,
            use_shared=False,
            shared_root_flag=True,
        ):
            pass

        class B(A, use_shared=True, shared_root_flag=False):
            pass

        class C(A, use_shared=False, shared_root_flag=False):
            pass

        assert A._get_id_counter() == 10
        assert A().unique_instance_id + 1 == A._get_id_counter()  # 11
        assert B().unique_instance_id + 1 == A._get_id_counter()  # 12
        assert B._get_id_counter() == 12

        assert C._get_id_counter() == 0
        assert C().unique_instance_id + 1 == C._get_id_counter()  # 0
        assert C().unique_instance_id + 1 == C._get_id_counter()  # 1
        assert C._get_id_counter() == 2


class TestSkipIfErrorContextManagerClass(object):
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
