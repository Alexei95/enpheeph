import typing

import pytorch_lightning
import torch
import torchmetrics


class PLVisionWrapper(pytorch_lightning.LightningModule):
    # we define the mapping for the optimizers
    # the name is the __qualname__ of the class mapped to the class itself
    # to get the correct optimizers, we search through the list of elements
    # in torch.optim, checking that the ones that have mro its result contains
    # torch.optim.Optimizer which is the base class for all optimizers
    OPTIMIZERS_DICT = {
            opt.__qualname__.lower(): opt
            for opt in dir(torch.optim)
            if hasattr(opt, 'mro') and torch.optim.Optimizer in opt.mro()
    }
    DEFAULT_OPTIMIZER_CLASS_NAME = 'adam'
    DEFAULT_OPTIMIZER_EXTRA_ARGS = {}
    DEFAULT_LEARNING_RATE = 1e-3
    # the default normalization function is softmax, and we compute it along
    # the last dimension as the first dimension is the batches, and we want
    # the results to be normalized across the elements in the batch
    DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION = torch.nn.Softmax(dim=-1)
    DEFAULT_LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    DEFAULT_ACCURACY_FUNCTION = torchmetrics.Accuracy()

    def __init__(
            self,
            model: torch.nn.Module,
            # this class should accept params and lr
            optimizer_class_name: str = DEFAULT_OPTIMIZER_CLASS_NAME,
            # extra arguments for the optimizer
            optimizer_extra_args:
            typing.Optional[
                    typing.Dict[str, typing.Any]
            ] = DEFAULT_OPTIMIZER_EXTRA_ARGS,
            lr: float = DEFAULT_LEARNING_RATE,
            *,
            # NOTE: fix an annoying bug with typing.Callable and jsonargparse
            # we should use the typing.Callable type hint but it doesn't work
            # normalize_prob_func: typing.Callable[
            #         [torch.Tensor],
            #         torch.Tensor,
            # ] = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION,
            normalize_prob_func:
            typing.Any = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION,
            # NOTE: fix an annoying bug with typing.Callable and jsonargparse
            # we should use the typing.Callable type hint but it doesn't work
            # loss_func: typing.Callable[
            #         [torch.Tensor, torch.Tensor],
            #         torch.Tensor,
            # ] = DEFAULT_LOSS_FUNCTION,
            loss_func: typing.Any = DEFAULT_LOSS_FUNCTION,
            # NOTE: fix an annoying bug with typing.Callable and jsonargparse
            # we should use the typing.Callable type hint but it doesn't work
            # accuracy_func: typing.Callable[
            #         [torch.Tensor, torch.Tensor],
            #         torch.Tensor,
            # ] = DEFAULT_ACCURACY_FUNCTION,
            accuracy_func: typing.Any = DEFAULT_ACCURACY_FUNCTION,
    ):
        super().__init__()

        self.model = model
        # if we get a KeyError for the optimizer name, we raise a ValueError
        # specifying the list of supported optimizers from the dict
        try:
            self.optimizer_class = self.OPTIMIZERS_DICT[optimizer_class_name]
        except KeyError:
            raise ValueError(
                    'Please use one of the supported optimizers: {}'.format(
                            tuple(self.OPTIMIZERS_DICT.keys())
                    )
            )
        self.optimizer_extra_args = optimizer_extra_args
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

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.parameters(), lr=self.lr, **self.optimizer_extra_args
        )
        return optimizer
