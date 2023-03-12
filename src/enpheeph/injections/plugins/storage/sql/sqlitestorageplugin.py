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

import typing

import sqlalchemy
import sqlalchemy.dialects.sqlite
import sqlalchemy.engine.url
import sqlalchemy.ext.compiler
import sqlalchemy.sql.expression
import sqlalchemy.types

import enpheeph.injections.plugins.storage.sql.abc.sqlstoragepluginabc
import enpheeph.injections.plugins.storage.sql.utils.sqlutils
import enpheeph.injections.plugins.storage.abc.storagepluginabc
import enpheeph.utils.dataclasses
import enpheeph.utils.typings

import enpheeph.injections.plugins.storage.sql.utils.sqldataclasses as sqldataclasses


class SQLiteStoragePlugin(
    # we disable black to avoid too long line issue in flake8
    # fmt: off
    (
        enpheeph.injections.plugins.storage.sql.abc.
        sqlstoragepluginabc.SQLStoragePluginABC
    ),
    # fmt: on
):
    DEFAULT_EXTRA_ENGINE_ARGS: typing.Dict[str, typing.Any] = {
        "future": True,
    }

    def __init__(
        self,
        db_url: str,
        # if True the SQLAlchemy engine prints all the queries in SQL
        # it is useful for debugging purposes
        extra_engine_args: typing.Dict[str, typing.Any] = DEFAULT_EXTRA_ENGINE_ARGS,
    ):
        # we generate the current engine
        # we set the current experiment id to None
        # NOTE: we use experiment id so that we can reload the experiment for each
        # new Session we create
        self.experiment_id: typing.Optional[int] = None
        self.session_id = None
        self.db_url = db_url
        self.extra_engine_args = extra_engine_args

        self.engine = self.init_engine(self.db_url, self.extra_engine_args)

    @classmethod
    def init_engine(
        cls,
        db_url: str,
        extra_engine_args: typing.Dict[str, typing.Any] = DEFAULT_EXTRA_ENGINE_ARGS,
    ) -> sqlalchemy.engine.Engine:
        # we create the engine
        engine = sqlalchemy.create_engine(db_url, **extra_engine_args)

        # we implement the fix if we are using pysqlite
        # to check, we get the dialect class from the url
        dialect: typing.Type[
            sqlalchemy.engine.Dialect
        ] = sqlalchemy.engine.url.make_url(db_url).get_dialect()
        # if pysqlite is in the dialect class name, we fix the engine for pysqlite
        if "pysqlite" in dialect.__qualname__:
            sqldataclasses.fix_pysqlite(engine)

        # we create all the tables in the engine
        sqldataclasses.CustomBase.metadata.create_all(engine)

        return engine
