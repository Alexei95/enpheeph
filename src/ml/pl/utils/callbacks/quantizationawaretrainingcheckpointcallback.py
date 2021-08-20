import typing

import pytorch_lightning


class QuantizationAwareTrainingCheckpointCallback(
        pytorch_lightning.callbacks.QuantizationAwareTraining
):
    def __init__(self, quantize_checkpoint: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._quantize_checkpoint = quantize_checkpoint

    def on_save_checkpoint(
            self,
            trainer,
            pl_module: pytorch_lightning.LightningModule,
            checkpoint: typing.Dict[str, typing.Any]
    ):
        if self._quantize_checkpoint:
            pass
