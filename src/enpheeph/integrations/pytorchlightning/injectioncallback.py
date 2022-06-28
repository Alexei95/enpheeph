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

import collections
import copy
import datetime
import typing
import warnings

import pytorch_lightning
import pytorch_lightning.callbacks

import enpheeph.handlers.injectionhandler
import enpheeph.injections.plugins.storage.abc.storagepluginabc

# to suppress all warnings
warnings.filterwarnings("ignore")


class InjectionCallback(pytorch_lightning.callbacks.Callback):
    experiment_time_start: typing.Optional[datetime.datetime]
    first_golden_run: typing.Union[bool, int]
    injection_handler: enpheeph.handlers.injectionhandler.InjectionHandler
    metrics: typing.DefaultDict[
        int, typing.DefaultDict[int, typing.DefaultDict[typing.Any, typing.Any]]
    ]
    metrics_save_frequency: typing.Optional[int]
    storage_plugin: typing.Optional[
        (enpheeph.injections.plugins.storage.abc.storagepluginabc.StoragePluginABC)
    ]
    test_epoch: int

    def __init__(
        self,
        injection_handler: (enpheeph.handlers.injectionhandler.InjectionHandler),
        storage_plugin: typing.Optional[
            (enpheeph.injections.plugins.storage.abc.storagepluginabc.StoragePluginABC)
        ] = None,
        # number of batches every which to save the metrics
        # additionally we save at the end of each epoch
        metrics_save_frequency: typing.Optional[int] = None,
        # if True, we use the first test run as golden run
        # otherwise, we expect it to be a valid id for the golden run reference
        first_golden_run: typing.Union[bool, int] = True,
        # extra session info
        extra_session_info: typing.Optional[typing.Dict[typing.Any, typing.Any]] = None,
        # extra experiment info which can be used to identify experiments
        extra_experiment_info: typing.Optional[
            typing.Dict[typing.Any, typing.Any]
        ] = None,
    ):
        self.experiment_time_start = None

        self.injection_handler = injection_handler
        self.storage_plugin = storage_plugin
        # this number is used to indicate how often to save the results
        # in terms of batch index
        self.metrics_save_frequency = metrics_save_frequency
        self.first_golden_run = first_golden_run

        self.extra_experiment_info = extra_experiment_info
        self.extra_session_info = extra_session_info

        self.test_epoch: int = 0
        # we use a defaultdict inside a defaultdict, so that when we access epoch, batch
        # we generate an empty dict
        # when we save this metric in the storage, it becomes a normal dict with
        # default_factory being reset to None
        self.metrics: typing.DefaultDict[
            int, typing.DefaultDict[int, typing.DefaultDict[typing.Any, typing.Any]]
        ] = collections.defaultdict(
            # mypy has issues with nested defaultdict
            lambda: collections.defaultdict(dict)  # type: ignore[arg-type]
        )

        # we create a new Session which will be closed on __del__
        self.storage_plugin.create_session(extra_session_info=extra_session_info)

    def __del__(self, *args, **kwargs):
        self.storage_plugin.complete_session()

        # not needed
        # super().__del__(*args, **kwargs)

    def on_test_start(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        self.test_epoch = 0
        self.metrics = collections.defaultdict(
            # mypy has issues with nested defaultdict
            lambda: collections.defaultdict(dict)  # type: ignore[arg-type]
        )

        self.injection_handler.setup(pl_module)

        # FIXME: use a MockStorage implementation
        # to allow this without checking for None
        if self.storage_plugin is not None:
            self.experiment_time_start = datetime.datetime.utcnow()
            self.storage_plugin.create_experiment(
                # we create an experiment with the active injections
                injection_locations=[
                    inj.location for inj in self.injection_handler.active_injections
                ],
                running=True,
                # we enable the golden run for the first execution only if the flag is
                # True
                golden_run_flag=self.first_golden_run is True,
                # we pass the id if the first_golden_run is an integer for the
                # experiment id
                # otherwise None to disable it
                golden_run_id=self.first_golden_run
                if isinstance(self.first_golden_run, int)
                else None,
                # we use UTC for dates as it is generic
                start_time=self.experiment_time_start,
                extra_experiment_info=self.extra_experiment_info,
            )

            # it will be True at most at the first iteration as we change it into int
            if self.first_golden_run is True:
                # casting as experiment_id is set, so it cannot be None
                experiment_id = typing.cast(int, self.storage_plugin.experiment_id)
                # we set the first_golden_run to the golden run id if the first test is
                # a golden run
                self.first_golden_run = experiment_id

    def on_test_end(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        self.save_metrics(trainer, test_epoch=-1, batch_idx=-1)

        self.test_epoch = 0

        if self.storage_plugin is not None:
            duration = (
                datetime.datetime.utcnow() - self.experiment_time_start
                if self.experiment_time_start is not None
                else None
            )
            self.storage_plugin.complete_experiment(
                total_duration=duration,
            )

            # we reset the start time
            self.experiment_time_start = None

        self.injection_handler.teardown(pl_module)

    def on_test_epoch_start(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        pass

    def on_test_epoch_end(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        self.save_metrics(trainer, test_epoch=self.test_epoch, batch_idx=-1)

        self.test_epoch += 1

    def on_test_batch_end(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
        outputs: typing.Optional[pytorch_lightning.utilities.types.STEP_OUTPUT],
        batch: typing.Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if (
            self.metrics_save_frequency is not None
            and not batch_idx % self.metrics_save_frequency
        ):
            self.save_metrics(trainer, test_epoch=self.test_epoch, batch_idx=batch_idx)

    def save_metrics(
        self,
        trainer: pytorch_lightning.Trainer,
        # we use -1 for the final result, can be substituted by globally
        # defined constant
        test_epoch: int,
        # we use -1 for the complete results at the end of the test
        # it could be substituted by a fixed constant in the future
        batch_idx: int,
    ) -> None:
        # if the storage_plugin is None, we skip all the computations
        if self.storage_plugin is not None:
            # we save the metrics only if the storage is available
            self.metrics[test_epoch][batch_idx] = copy.deepcopy(
                # mypy has issues with nested defaultdict
                # we need to save all the metrics, with progress bar < callback < logged
                {
                    **trainer.progress_bar_metrics,
                    **trainer.callback_metrics,
                    **trainer.logged_metrics,
                }
            )

            self.metrics[test_epoch][batch_idx] = {
                k: v.item() for k, v in self.metrics[test_epoch][batch_idx].items()
            }

            # we copy the metrics, so we can change the defaultdict behaviour without
            # changing the original
            metrics = copy.deepcopy(self.metrics)

            # we remove all the default factories so that a missing key gives KeyError
            metrics.default_factory = None
            for el in metrics.values():
                el.default_factory = None

            self.storage_plugin.add_experiment_metrics(metrics)
