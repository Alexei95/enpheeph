import typing

import pytorch_lightning
import torch
import torchmetrics


class PLVisionWrapper(
        pytorch_lightning.LightningModule,
):
    DEFAULT_OPTIMIZER_CLASS = torch.optim.Adam
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_BATCH_SIZE = 1
    # the default normalization function is softmax, and we compute it along
    # the last dimension as the first dimension is the batches, and we want
    # the results to be normalized across the elements in the batch
    DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION = torch.nn.Softmax(dim=-1)
    DEFAULT_LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    DEFAULT_ACCURACY_FUNCTION = torchmetrics.Accuracy()

    def __init__(
            self,
            model: torch.nn.Module,
            learning_rate: float = DEFAULT_LEARNING_RATE,
            batch_size: int = DEFAULT_BATCH_SIZE,
            *,
            # this class should accept params and lr
            optimizer_class: typing.Callable[
                        [typing.Iterable[torch.nn.parameter.Parameter], float],
                        torch.optim.Optimizer,
            ] = DEFAULT_OPTIMIZER_CLASS,
            normalize_prob_func: typing.Callable[
                        [torch.Tensor],
                        torch.Tensor,
            ] = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION,
            loss_func: typing.Callable[
                        [torch.Tensor, torch.Tensor],
                        torch.Tensor,
            ] = DEFAULT_LOSS_FUNCTION,
            accuracy_func: typing.Callable[
                        [torch.Tensor, torch.Tensor],
                        torch.Tensor,
            ] = DEFAULT_ACCURACY_FUNCTION,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self.hparams.model
        self.optimizer_class = self.hparams.optimizer_class

        # we keep lr in the model to allow for Trainer.tune
        # to run and determine the optimal ones
        self.learning_rate = self.hparams.learning_rate
        # same for batch size
        self.batch_size = self.hparams.batch_size

        self.normalize_prob_func = self.hparams.normalize_prob_func
        self.loss_func = self.hparams.loss_func
        self.accuracy_func = self.hparams.accuracy_func

    def forward(self, input_):
        return self.model(input_)

    # implemented by us for compatibility between forward and validation/test
    # steps
    def inference_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.normalize_prob_func(self.forward(x))
        loss = self.loss_func(y_hat, y)
        acc = self.accuracy_func(y_hat, y)

        return {'acc': acc, 'loss': loss}

    def training_step(self, batch, batch_idx):
        m = self.inference_step(batch, batch_idx)
        metrics = {
                'train_acc': m['acc'],
                'train_loss': m['loss'],
        }
        self.log_dict(
                metrics,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
        )
        # here we need to return the loss to be able to properly train
        return m['loss']

    def validation_step(self, batch, batch_idx):
        m = self.inference_step(batch, batch_idx)
        metrics = {
                'val_acc': m['acc'],
                'val_loss': m['loss'],
        }
        self.log_dict(
                metrics,
                prog_bar=True,
                on_step=True,
                on_epoch=True
        )
        # this may not be needed, as for logging we already use self.log_dict
        # return metrics

    def test_step(self, batch, batch_idx):
        m = self.inference_step(batch, batch_idx)
        metrics = {
                'test_acc': m['acc'],
                'test_loss': m['loss'],
        }
        self.log_dict(
                metrics,
                prog_bar=True,
                on_step=True,
                on_epoch=True
        )
        # this may not be needed, as for logging we already use self.log_dict
        # return metrics

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        return optimizer
