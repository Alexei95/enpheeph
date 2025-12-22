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

import abc
import datetime
import typing

import enpheeph.injections.plugins.storage.utils.storagetypings
import enpheeph.utils.dataclasses


class StoragePluginABC(abc.ABC):
    # the id of the current experiment
    experiment_id: typing.Optional[int]
    session_id: typing.Optional[int]

    @abc.abstractmethod
    def get_experiments(
        self,
        id_: typing.Optional[int] = None,
        running: typing.Optional[bool] = None,
        completed: typing.Optional[bool] = None,
        start_time: typing.Optional[datetime.datetime] = None,
        total_duration: typing.Optional[datetime.timedelta] = None,
        golden_run_flag: typing.Optional[bool] = None,
        injection_locations: typing.Optional[
            typing.Sequence[enpheeph.utils.dataclasses.InjectionLocationABC]
        ] = None,
        # in the future we will add also model_info
    ) -> typing.List[
        enpheeph.injections.plugins.storage.utils.storagetypings.ExperimentRunProtocol,
    ]:
        pass

    @abc.abstractmethod
    def create_experiment(
        self,
        injection_locations: typing.Sequence[
            enpheeph.utils.dataclasses.InjectionLocationABC
        ],
        # in the future also model_info
        running: bool = True,
        golden_run_flag: bool = False,
        # the id for the golden run
        # if None we skip this part
        golden_run_id: typing.Optional[int] = None,
        start_time: typing.Optional[datetime.datetime] = None,
        extra_experiment_info: typing.Optional[
            typing.Dict[typing.Any, typing.Any]
        ] = None,
    ) -> int:
        pass

    @abc.abstractmethod
    def create_session(
        self,
        extra_session_info: typing.Optional[typing.Dict[typing.Any, typing.Any]] = None,
    ) -> int:
        pass

    @abc.abstractmethod
    def complete_experiment(
        self,
        total_duration: typing.Optional[datetime.timedelta] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def complete_session(
        self,
    ) -> None:
        pass

    @abc.abstractmethod
    def add_experiment_metrics(
        self, metrics: typing.Dict[typing.Any, typing.Any]
    ) -> None:
        pass

    @abc.abstractmethod
    def add_experiment_golden_run(self, golden_run_id: int) -> None:
        pass

    @abc.abstractmethod
    def add_payload(
        self,
        location: enpheeph.utils.dataclasses.InjectionLocationABC,
        payload: typing.Dict[typing.Any, typing.Any],
    ) -> None:
        pass
