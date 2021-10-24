# -*- coding: utf-8 -*-
import typing

import sqlalchemy
import sqlalchemy.dialects.postgresql
import sqlalchemy.ext.compiler
import sqlalchemy.sql.expression
import sqlalchemy.types

import enpheeph.injections.plugins.sqlstorageplugin.sqldatastructure
import enpheeph.injections.plugins.sqlstorageplugin.sqlutils
import enpheeph.injections.plugins.storagepluginabc
import enpheeph.utils.typings


class PostgreSQLStoragePlugin(
    enpheeph.injections.plugins.storagepluginabc.StoragePluginABC,
):
    def __init__(
        self, db_url: str,
    ):
        # we add the DB + API
        self.db_url = "postgresql+psycopg2://" + db_url

        self.engine = sqlalchemy.engine(
            self.db_url,
            future=True,
            # we setup the highest isolation to avoid concurrent reads/
            # writes, to prepare for multiprocessing
            execution_options={"isolation_level": "SERIALIZABLE"},
        )

    def add_element(self, element_name: str, element: typing.Any) -> None:
        self.current_dict[element_name] = copy.deepcopy(element)

    def add_dict(self, dict_: typing.Dict[str, typing.Any]) -> None:
        self.current_dict.update({key: copy.deepcopy(value) for key, value in dict_})

        if self.auto_submit_eol:
            self.submit_eol()
        # if auto_submit_eol is True then the auto_execute is delegated in
        # submit_eol, otherwise it must be checked here
        elif self.auto_execute:
            self.execute()

    def submit_eol(self) -> None:
        if self.current_dict or self.work_on_empty:
            self.list_of_dicts.append(self.current_dict)

            self.current_dict = {}

        if self.auto_execute:
            self.execute()

    def execute(self) -> None:
        if not any(self.list_of_dicts) and not self.work_on_empty:
            return

        self.pandas_df = pandas.DataFrame(self.list_of_dicts)

        if self.dask_df is None:
            self.dask_df = dask.dataframe.from_pandas(
                self.pandas_df, chunksize=self.dask_chunksize,
            )
        else:
            self.dask_df = self.dask_df.append(self.pandas_df, ignore_index=True)

        self.dask_df.to_parquet(path=self.path, **self.dask_parquet_config)

        self.list_of_dicts = []
