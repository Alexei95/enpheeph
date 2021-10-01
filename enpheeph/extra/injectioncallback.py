import pytorch_lightning

import enpheeph.handlers.injectionhandler


class InjectionCallback(pytorch_lightning.callbacks.Callback):
    def __init__(
            self,
            injection_manager: enpheeph.handlers.injectionhandler.
            InjectionHandler
    ):
        self.injection_manager = injection_manager

    def on_test_start(
            self,
            trainer: pytorch_lightning.Trainer,
            pl_module: pytorch_lightning.LightningModule,
    ):
        self.injection_manager.setup(pl_module)

    def on_test_end(self,
            trainer: pytorch_lightning.Trainer,
            pl_module: pytorch_lightning.LightningModule,
    ):
        self.injection_manager.teardown(pl_module)
