import typing

import pytorch_lightning


class PruningSparsifyCallback(pytorch_lightning.callbacks.ModelPruning):
    def __init__(self, make_sparse: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._make_sparse = make_sparse

    def on_train_end(
            self,
            trainer,
            pl_module: pytorch_lightning.LightningModule
    ):
        super().on_train_end(trainer, pl_module)

        if self._make_sparse:
            pass

    def on_save_checkpoint(
            self,
            trainer,
            pl_module: pytorch_lightning.LightningModule,
            checkpoint: typing.Dict[str, typing.Any]
    ):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        if self._make_sparse:
            pass
