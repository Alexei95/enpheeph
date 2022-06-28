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

import datetime
import dataclasses
import typing


@dataclasses.dataclass(init=True, repr=True, eq=True, order=True)
class ExperimentRun(object):
    id_: int
    running: bool = False
    completed: bool = False
    start_time: typing.Optional[datetime.datetime] = None
    total_duration: typing.Optional[datetime.timedelta] = None
    golden_run_flag: bool = False
    metrics: typing.Optional[typing.Dict[str, typing.Any]] = None

    polymorphic_discriminator = None

    injections: typing.Optional[typing.Sequence["Injection"]] = None

    golden_run: typing.Optional["ExperimentRun"] = None
    golden_run_id: typing.Optional[int] = None

    injected_runs: typing.Optional[typing.Sequence["ExperimentRun"]] = None


@dataclasses.dataclass(init=True, repr=True, eq=True, order=True)
class Injection(object):
    location: typing.Any

    internal_id: int

    experiment_run_id: typing.Optional[int] = None
    experiment_run: typing.Optional["ExperimentRun"] = None


@dataclasses.dataclass(init=True, repr=True, eq=True, order=True)
class Fault(Injection):
    pass


@dataclasses.dataclass(init=True, repr=True, eq=True, order=True)
class Monitor(Injection):
    payload: typing.Optional[typing.Dict[str, typing.Any]] = None
