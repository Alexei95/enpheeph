# -*- coding: utf-8 -*-
import copy
import pathlib
import typing

import dask.dataframe
import pandas

import enpheeph.injections.plugins.storagepluginabc
import enpheeph.utils.typings


# we use Dask as temporary storage
class DaskStoragePlugin(enpheeph.injections.plugins.storagepluginabc.StoragePluginABC,):
    def __init__(
        self,
        path: enpheeph.utils.typings.PathType,
        # extra arguments for the to_parquet function in dask DataFrame
        dask_parquet_config: typing.Dict[str, typing.Any],
        # number of rows to keep in each partition in dask DataFrame
        dask_chunksize: int = 1024,
    ):
        self.path = pathlib.Path(path)

        self.dask_chunksize = dask_chunksize

        self.dask_parquet_config = dask_parquet_config

        self.dask_df = None
        self.pandas_df = None

        self.list_of_dicts = []
        self.current_dict = {}

    def add_element(self, element_name: str, element: typing.Any) -> None:
        self.current_dict[element_name] = copy.deepcopy(element)

    def add_dict(self, dict_: typing.Dict[str, typing.Any]) -> None:
        self.current_dict.update({key: copy.deepcopy(value) for key, value in dict_})

    def submit_eol(self) -> None:
        if self.current_dict:
            self.list_of_dicts.append(self.current_dict)

            self.current_dict = {}

    def execute(self) -> None:
        if not any(self.list_of_dicts):
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
