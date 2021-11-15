# -*- coding: utf-8 -*-
import collections
import copy
import typing

import pytorch_lightning

import enpheeph.handlers.injectionhandler
import enpheeph.injections.plugins.storage.storagepluginabc


class InjectionCallback(pytorch_lightning.callbacks.Callback):
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
            int, typing.DefaultDict[int, typing.DefaultDict[str, typing.Any]]
        ] = collections.defaultdict(lambda: collections.defaultdict(dict))

    def on_test_start(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        self.test_epoch = 0
        self.metrics = collections.defaultdict(lambda: collections.defaultdict(dict))

        self.injection_handler.setup(pl_module)

        # FIXME: use a MockStorage implementation
        # to allow this without checking for None
        if self.storage_plugin is not None:
            self.storage_plugin.create_experiment(
                # we create an experiment with the active injections
                injection_locations=[
                    inj.location for inj in self.injection_handler.active_injections
                ],
                running=True,
                golden_run_flag=self.first_golden_run is True,
                golden_run_id=self.first_golden_run
                if isinstance(self.first_golden_run, int)
                else None,
            )

    def on_test_end(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        self.save_metrics()

        self.test_epoch = 0

        if self.storage_plugin is not None:
            self.storage_plugin.complete_experiment()

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
