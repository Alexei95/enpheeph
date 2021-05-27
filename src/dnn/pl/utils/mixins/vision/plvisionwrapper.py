import typing

import pytorch_lightning
import torch
import torchmetrics


DEFAULT_OPTIMIZER_CLASS = torch.optim.Adam
DEFAULT_LEARNING_RATE = 1e-3
# the default normalization function is softmax, and we compute it along the
# last dimension as the first dimension is the batches, and we want the results
# to be normalized across the elements in the batch
DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION = torch.nn.Softmax(dim=-1)
DEFAULT_LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
DEFAULT_ACCURACY_FUNCTION = torchmetrics.Accuracy()


class PLVisionWrapper(pytorch_lightning.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            # this class should accept params and lr
            # if a custom implementation is required, i.e. a custom beta1 and
            # beta2 for Adam, use functools.partial
            optimizer_class: typing.Callable[
                    [typing.Iterable, float],
                    torch.optim.Optimizer
            ] = DEFAULT_OPTIMIZER_CLASS,
            lr: float = DEFAULT_LEARNING_RATE,
            *,
            normalize_prob_func: typing.Callable[
                    [torch.Tensor],
                    torch.Tensor,
            ] = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION,
            loss_func: typing.Callable[
                    [torch.Tensor, torch.Tensor],
                    torch.Tensor,
            ] = DEFAULT_LOSS_FUNCTION,
            # accuracy_func: typing.Callable[
            #         [torch.Tensor, torch.Tensor],
            #         torch.Tensor,
            # ] = DEFAULT_ACCURACY_FUNCTION,
            # NOTE: fix an annoying bug with typing.Callable and jsonargparse
            # we should use the typing.Callable type hint but it doesn't work
            accuracy_func: typing.Any = DEFAULT_ACCURACY_FUNCTION,
    ):
        super().__init__()

        self.model = model
        self.optimizer_class = optimizer_class
        # we keep lr in the model to allow for Trainer.tune
        # to run and determine the optimal ones
        self.lr = lr
        self.accuracy_func = accuracy_func
        self.loss_func = loss_func
        self.normalize_prob_func = normalize_prob_func

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

    def configure_optimizer(self):
        optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr)
        return optimizer
