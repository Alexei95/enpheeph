# -*- coding: utf-8 -*-
import typing

import pytorch_lightning

import enpheeph.handlers.injectionhandler
import enpheeph.injections.plugins.storagepluginabc


class InjectionCallback(pytorch_lightning.callbacks.Callback):
    def __init__(
        self,
        injection_manager: (enpheeph.handlers.injectionhandler.InjectionHandler),
        storage_plugin: typing.Optional[
            (enpheeph.injections.plugins.storagepluginabc.StoragePluginABC)
        ],
    ):
        self.injection_manager = injection_manager
        self.storage_plugin = storage_plugin

        self.test_epoch = 0

    def on_test_start(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        self.test_epoch = 0

        self.injection_manager.setup(pl_module)

    def on_test_end(
        self,
        trainer: pytorch_lightning.Trainer,
        pl_module: pytorch_lightning.LightningModule,
    ) -> None:
        self.test_epoch = 0

        self.injection_manager.teardown(pl_module)

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
        self.storage_plugin.execute()

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
        self.storage_plugin.add_element("test_epoch", self.test_epoch)
        self.storage_plugin.add_element("test_batch", batch_idx)
        self.storage_plugin.submit_eol()
