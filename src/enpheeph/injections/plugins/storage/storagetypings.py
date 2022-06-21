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
import typing


# NOTE: we use typing.Protocol as it is quite difficult to make abc.ABC work with
# SQLAlchemy, so in this way it is easier to use for the different
@typing.runtime_checkable
class ExperimentRunProtocol(typing.Protocol):
    id_: int
    running: bool
    completed: bool
    start_time: typing.Optional[datetime.datetime]
    total_duration: typing.Optional[datetime.timedelta]
    golden_run_flag: bool
    metrics: typing.Optional[typing.Dict[str, typing.Any]]

    polymorphic_discriminator: typing.Optional[str]

    injections: typing.Optional[typing.Sequence["InjectionProtocol"]]

    golden_run: typing.Optional["ExperimentRunProtocol"]
    golden_run_id: typing.Optional[int]

    injected_runs: typing.Optional[typing.Sequence["ExperimentRunProtocol"]]


@typing.runtime_checkable
class InjectionProtocol(typing.Protocol):
    location: typing.Any

    internal_id: int

    experiment_run_id: typing.Optional[int]
    experiment_run: typing.Optional["ExperimentRunProtocol"]


@typing.runtime_checkable
class FaultProtocol(InjectionProtocol, typing.Protocol):
    pass


@typing.runtime_checkable
class MonitorProtocol(InjectionProtocol, typing.Protocol):
    payload: typing.Optional[typing.Dict[str, typing.Any]]
