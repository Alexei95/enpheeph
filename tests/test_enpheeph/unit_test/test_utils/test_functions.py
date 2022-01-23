# -*- coding: utf-8 -*-
import pytest

import enpheeph.utils.functions


class TestFunctions(object):
    @pytest.mark.parametrize(
        argnames=("camel", "snake"),
        argvalues=[
            pytest.param(
                "CamelSnake",
                "camel_snake",
                id="CamelSnake",
            ),
            pytest.param(
                "camelSnake",
                "camel_snake",
                id="camelSnake",
            ),
            pytest.param(
                "camel_snake",
                "camel_snake",
                id="camel_snake",
            ),
        ],
    )
    def test_camel_to_snake(self, camel, snake):
        assert enpheeph.utils.functions.camel_to_snake(camel) == snake
