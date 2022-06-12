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

import typing


def config(
    *args: typing.Any,
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    return {
        "injection_config": {
            # the list contains the percentage of injections
            # we cover 10 elements per decade, and add 1 at the end and 0 at the start
            # we start from 0.000001
            # "randomness": [0] + sum((list(x * 10 ** y for x in range(1, 10)) for y in range(-7, 0)), start=[]) + [1],
            "randomness": [
                0.003,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009000000000000001,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.1,
                0.2,
                0.30000000000000004,
                0.4,
                0.5,
                0.6000000000000001,
                0.7000000000000001,
                0.8,
                0.9,
                1,
            ],
            # custom set to false to allow the random injections to be instantiated
            "custom": False,
        }
    }
