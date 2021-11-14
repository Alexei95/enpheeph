# -*- coding: utf-8 -*-
import abc
import datetime
import typing

import enpheeph.injections.plugins.storage.storage_typings
import enpheeph.utils.data_classes


class StoragePluginABC(abc.ABC):
    @abc.abstractmethod
    def get_experiments(
        self,
        id_: typing.Optional[int] = None,
        running: typing.Optional[bool] = None,
        completed: typing.Optional[bool] = None,
        start_time: typing.Optional[datetime.datetime] = None,
        total_duration: typing.Optional[float] = None,
        golden_run_flag: typing.Optional[bool] = None,
        injection_locations: typing.Optional[
            typing.Sequence[enpheeph.utils.data_classes.InjectionLocationABC]
        ] = None,
        # in the future we will add also model_info
    ) -> typing.Sequence[
        enpheeph.injections.plugins.storage.storage_typings.ExperimentRunProtocol,
    ]:
        pass

    @abc.abstractmethod
    def create_experiment(
        self,
        injection_locations: typing.Sequence[
            enpheeph.utils.data_classes.InjectionLocationABC
        ],
        # in the future also model_info
        running: bool = True,
        golden_run_flag: bool = False,
        # the id for the golden run
        # if None we skip this part
        golden_run_id: typing.Optional[bool] = None,
    ) -> int:
        pass

    @abc.abstractmethod
    def complete_experiment(
        self,
        start_time: typing.Optional[datetime.datetime],
        total_duration: typing.Optional[float] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def add_experiment_metrics(self, metrics: typing.Dict[str, typing.Any]) -> None:
        pass

    @abc.abstractmethod
    def add_experiment_golden_run(self, golden_run_id: int) -> None:
        pass

    @abc.abstractmethod
    def add_payload(
        self,
        location: enpheeph.utils.data_classes.InjectionLocationABC,
        payload: typing.Dict[str, typing.Any],
    ) -> None:
        pass
