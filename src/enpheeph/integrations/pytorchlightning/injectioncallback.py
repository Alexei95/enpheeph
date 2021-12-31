# -*- coding: utf-8 -*-
import collections
import copy
import datetime
import typing
import warnings

import pytorch_lightning

import enpheeph.handlers.injectionhandler
import enpheeph.injections.plugins.storage.storagepluginabc

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
        (enpheeph.injections.plugins.storage.storagepluginabc.StoragePluginABC)
    ]
    test_epoch: int

    def __init__(
        self,
        injection_handler: (enpheeph.handlers.injectionhandler.InjectionHandler),
        storage_plugin: typing.Optional[
            (enpheeph.injections.plugins.storage.storagepluginabc.StoragePluginABC)
        ] = None,
        # number of batches every which to save the metrics
        # additionally we save at the end of each epoch
        metrics_save_frequency: typing.Optional[int] = None,
        # if True, we use the first test run as golden run
        # otherwise, we expect it to be a valid id for the golden run reference
        first_golden_run: typing.Union[bool, int] = True,
    ):
        self.experiment_time_start = None

        self.injection_handler = injection_handler
        self.storage_plugin = storage_plugin
        # this number is used to indicate how often to save the results
        # in terms of batch index
        self.metrics_save_frequency = metrics_save_frequency
        self.first_golden_run = first_golden_run

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
            )

            # it will be True at most at the first iteration as we change it into int
            if self.first_golden_run is True:
                # we set the first_golden_run to the golden run id if the first test is
                # a golden run
                self.first_golden_run = self.storage_plugin.experiment_id

    def on_test_end(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        self.save_metrics()

        self.test_epoch = 0

        if self.storage_plugin is not None:
            duration = (
                datetime.datetime.utcnow() - self.experiment_time_start
                if self.experiment_time_start is not None
                else None
            )
            self.experiment_time_start = None
            self.storage_plugin.complete_experiment(
                total_duration=duration,
            )

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
        self.save_metrics()

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
        self.metrics[self.test_epoch][batch_idx] = copy.deepcopy(
            # mypy has issues with nested defaultdict
            trainer.callback_metrics
        )

        if (
            self.metrics_save_frequency is not None
            and not batch_idx % self.metrics_save_frequency
        ):
            self.save_metrics()

    def save_metrics(self) -> None:
        # if the storage_plugin is None, we skip all the computations
        if self.storage_plugin is not None:
            # we copy the metrics, so we can change the defaultdict behaviour without
            # changing the original
            metrics = copy.deepcopy(self.metrics)

            # we remove all the default factories so that a missing key gives KeyError
            metrics.default_factory = None
            for el in metrics.values():
                el.default_factory = None

            self.storage_plugin.add_experiment_metrics(metrics)
