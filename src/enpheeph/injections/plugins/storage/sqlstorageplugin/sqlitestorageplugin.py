# -*- coding: utf-8 -*-
import typing

import sqlalchemy
import sqlalchemy.dialects.sqlite
import sqlalchemy.engine.url
import sqlalchemy.ext.compiler
import sqlalchemy.sql.expression
import sqlalchemy.types

import enpheeph.injections.plugins.storage.sqlstorageplugin.sqlstoragepluginabc
import enpheeph.injections.plugins.storage.sqlstorageplugin.sqlutils
import enpheeph.injections.plugins.storage.storagepluginabc
import enpheeph.utils.data_classes
import enpheeph.utils.typings

from enpheeph.injections.plugins.storage.sqlstorageplugin import sql_data_classes


class SQLiteStoragePlugin(
    # we disable black to avoid too long line issue in flake8
    # fmt: off
    (
        enpheeph.injections.plugins.storage.sqlstorageplugin.
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
            sql_data_classes.fix_pysqlite(engine)

        # we create all the tables in the engine
        sql_data_classes.CustomBase.metadata.create_all(engine)

        return engine
