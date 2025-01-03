# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2023 Alessio "Alexei95" Colucci
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

import sqlalchemy
import sqlalchemy.dialects.sqlite
import sqlalchemy.ext.compiler
import sqlalchemy.sql.expression
import sqlalchemy.types

import enpheeph.injections.plugins.storage.sql.utils.sqlutils
import enpheeph.injections.plugins.storage.abc.storagepluginabc
import enpheeph.injections.plugins.storage.utils.storagetypings
import enpheeph.utils.dataclasses
import enpheeph.utils.typings

import enpheeph.injections.plugins.storage.sql.utils.sqldataclasses as sql_data_classes


class SQLStoragePluginABC(
    enpheeph.injections.plugins.storage.abc.storagepluginabc.StoragePluginABC,
):
    experiment_id: typing.Optional[int]
    engine: sqlalchemy.engine.Engine
    session_id: typing.Optional[int]

    @classmethod
    @abc.abstractmethod
    def init_engine(
        cls, db_url: str, extra_engine_args: typing.Dict[str, typing.Any]
    ) -> sqlalchemy.engine.Engine:
        pass

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
        # first we open the querying session on our engine
        with sqlalchemy.orm.Session(self.engine) as session:
            # this is the statement for selecting the ExperimentRun
            stmt = sqlalchemy.select(sql_data_classes.ExperimentRun)

            # filtering by attributes is easily doable with a query on the properties

            if id_ is not None:
                stmt = stmt.where(sql_data_classes.ExperimentRun.id_ == id_)

            if running is not None:
                stmt = stmt.where(sql_data_classes.ExperimentRun.running == running)

            if completed is not None:
                stmt = stmt.where(sql_data_classes.ExperimentRun.completed == completed)

            if start_time is not None:
                stmt = stmt.where(
                    sql_data_classes.ExperimentRun.start_time == start_time
                )

            if total_duration is not None:
                stmt = stmt.where(
                    sql_data_classes.ExperimentRun.total_duration == total_duration
                )

            if golden_run_flag is not None:
                stmt = stmt.where(
                    sql_data_classes.ExperimentRun.golden_run_flag == golden_run_flag
                )

            # if we filter by injection locations
            if injection_locations is not None:
                # we need to create the aliases,
                # one for each injection in the input list
                inj_aliases = [
                    sqlalchemy.orm.aliased(sql_data_classes.Injection)
                    for _ in range(len(injection_locations))
                ]
                # then we join the aliases,
                # so that each one of them is bound to a different
                # instance of Injection that are connected to ExperimentRun
                for inj_al in inj_aliases:
                    stmt = stmt.join(inj_al, sql_data_classes.ExperimentRun.injections)
                # then we add the conditions
                for inj_al, inj in zip(inj_aliases, injection_locations):
                    stmt = stmt.where(inj_al.location == inj)

            # we return all the instances of the classes
            return typing.cast(
                typing.List[
                    # black converts it to very long line, so we disable it
                    # fmt: off
                    enpheeph.injections.plugins.storage.
                    storage_typings.ExperimentRunProtocol,
                    # fmt: on
                ],
                session.execute(stmt).scalars().all(),
            )

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
        # check to avoid creating an experiment on top of the existing one
        if self.experiment_id is not None:
            raise ValueError("To create an experiment the current one must be closed")

        if self.session_id is None:
            raise ValueError(
                "To create an experiment you need to create a Session first"
            )

        # we open a new Session
        with sqlalchemy.orm.Session(self.engine) as session:
            # we create a new ExperimentRun
            # running is set by default, but the argument is_running can disable it
            experiment = sql_data_classes.ExperimentRun(
                running=running,
                completed=False,
                golden_run_flag=golden_run_flag,
                start_time=start_time,
                extra_experiment_info=extra_experiment_info,
                # we set the ID for the Session the experiment is on
                session_id=self.session_id,
            )

            # we insert all the injection locations
            # depending on the class instance we create different objects
            for inj_loc in injection_locations:
                if isinstance(inj_loc, enpheeph.utils.dataclasses.FaultLocation):
                    sql_inj_loc = sql_data_classes.Fault(
                        location=inj_loc, internal_id=inj_loc.unique_instance_id
                    )
                elif isinstance(inj_loc, enpheeph.utils.dataclasses.MonitorLocation):
                    sql_inj_loc = sql_data_classes.Monitor(
                        location=inj_loc, internal_id=inj_loc.unique_instance_id
                    )
                else:
                    raise ValueError(f"Unsupported injection, {inj_loc}")
                experiment.injections.append(sql_inj_loc)

            session.add(experiment)

            session.commit()

            # ID is available only after committing
            self.experiment_id = experiment.id_

        # this must be done outside the session, as experiment is not added yet
        # to the database
        if golden_run_id is not None:
            self.add_experiment_golden_run(golden_run_id)

        return self.experiment_id

    def create_session(
        self,
        extra_session_info: typing.Optional[typing.Dict[typing.Any, typing.Any]] = None,
    ) -> int:
        # check to avoid creating an experiment on top of the existing one
        if self.experiment_id is not None:
            raise ValueError(
                "To create a Session the current experiment must be closed"
            )
        if self.session_id is not None:
            raise ValueError("To create a Session the current one must be closed")

        # we open a new Session
        with sqlalchemy.orm.Session(self.engine) as session:
            # we create a new Session
            sess = sql_data_classes.Session(
                extra_session_info=extra_session_info,
            )

            session.add(sess)

            session.commit()

            # ID is available only after committing
            self.session_id = sess.id_

        return self.session_id

    def complete_experiment(
        self,
        total_duration: typing.Optional[datetime.timedelta] = None,
    ) -> None:
        if self.experiment_id is None:
            raise ValueError("There is no experiment to be closed")

        with sqlalchemy.orm.Session(self.engine) as session:
            # we get the experiment from the session
            experiment = (
                session.execute(
                    sqlalchemy.select(sql_data_classes.ExperimentRun).where(
                        sql_data_classes.ExperimentRun.id_ == self.experiment_id
                    )
                )
                .scalars()
                .one()
            )  # we use .one() as there will be only one match

            experiment.completed = True
            experiment.total_duration = total_duration

            experiment.running = False

            session.add(experiment)

            session.commit()

        self.experiment_id = None

    def complete_session(
        self,
    ) -> None:
        if self.experiment_id is not None:
            raise ValueError(
                "To close the current Session the current Experiment must be closed"
            )
        if self.session_id is None:
            raise ValueError("There is no current Session to close")

        with sqlalchemy.orm.Session(self.engine) as session:
            # we get the Session from the engine
            sess = (
                session.execute(
                    sqlalchemy.select(sql_data_classes.Session).where(
                        sql_data_classes.Session.id_ == self.session_id
                    )
                )
                .scalars()
                .one()
            )  # we use .one() as there will be only one match

            # **NOTE**: for now we do not do anything but it might be useful in the
            # future
            # ideally the Session should represent all the information which are common
            # between Experiments from the same run, but ideally different experiments
            # in the same run might also have different models/datasets, so more
            # details are needed

            session.add(sess)

            session.commit()

        self.session = None

    def add_experiment_metrics(
        self, metrics: typing.Dict[typing.Any, typing.Any]
    ) -> None:
        if self.experiment_id is None:
            raise ValueError("There is no experiment to be closed")

        with sqlalchemy.orm.Session(self.engine) as session:
            # we get the experiment from the session
            experiment = (
                session.execute(
                    sqlalchemy.select(sql_data_classes.ExperimentRun).where(
                        sql_data_classes.ExperimentRun.id_ == self.experiment_id
                    )
                )
                .scalars()
                .one()
            )  # we use .one() as there will be only one match

            experiment.metrics = metrics

            session.add(experiment)

            session.commit()

    def add_experiment_golden_run(self, golden_run_id: int) -> None:
        if self.experiment_id is None:
            raise ValueError("There is no experiment to work on")

        with sqlalchemy.orm.Session(self.engine) as session:
            # we get the experiment from the session
            experiment = (
                session.execute(
                    sqlalchemy.select(sql_data_classes.ExperimentRun).where(
                        sql_data_classes.ExperimentRun.id_ == self.experiment_id
                    )
                )
                .scalars()
                .one()
            )  # we use .one() as there will be only one match

            # we cannot get the golden run directly as that would be a circular
            # dependency
            # so we simply update the ID
            experiment.golden_run_id = golden_run_id

            session.add(experiment)

            session.commit()

    def add_payload(
        self,
        location: enpheeph.utils.dataclasses.InjectionLocationABC,
        payload: typing.Dict[typing.Any, typing.Any],
    ) -> None:
        # we create a new session on the engine
        with sqlalchemy.orm.Session(self.engine) as session:
            # we query the session to get the corresponding element from the current
            # experiment
            stmt = (
                sqlalchemy.select(sql_data_classes.Injection)
                .select_from(sql_data_classes.ExperimentRun)
                .join(sql_data_classes.Injection)
                .where(
                    sql_data_classes.Injection.experiment_run_id == self.experiment_id
                )
                .where(sql_data_classes.Injection.location == location)
                .where(
                    sql_data_classes.Injection.internal_id
                    == location.unique_instance_id
                )
            )

            inj = session.execute(stmt).scalars().one()

            inj.payload = payload

            session.add(inj)

            session.commit()
