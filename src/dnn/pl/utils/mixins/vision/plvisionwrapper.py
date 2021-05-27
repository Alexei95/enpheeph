import typing

import pytorch_lightning
import torch
import torchmetrics

import src.utils.mixins.subclassgatherer


class PLVisionWrapper(
        pytorch_lightning.LightningModule,
        src.utils.mixins.subclassgatherer.SubclassGatherer
):
    OPTIMIZERS_DICT = src.utils.mixins.\
    subclassgatherer.SubclassGatherer.gather_subclasses(
            module=torch.optim,
            baseclass=torch.optim.Optimizer,
    )
    NORMALIZATIONS_DICT = src.utils.mixins.\
    subclassgatherer.SubclassGatherer.gather_subclasses(
            module=torch.nn.modules.activation,
            baseclass=torch.nn.Module,
    )
    LOSSES_DICT = src.utils.mixins.\
    subclassgatherer.SubclassGatherer.gather_subclasses(
            module=torch.nn.modules.loss,
            baseclass=torch.nn.Module,
    )
    ACCURACIES_DICT = src.utils.mixins.\
    subclassgatherer.SubclassGatherer.gather_subclasses(
            module=torchmetrics,
            baseclass=torchmetrics.metric.Metric,
    )

    DEFAULT_OPTIMIZER_CLASS_NAME = 'adam'
    DEFAULT_OPTIMIZER_EXTRA_ARGS = {}
    DEFAULT_LEARNING_RATE = 1e-3
    # the default normalization function is softmax, and we compute it along
    # the last dimension as the first dimension is the batches, and we want
    # the results to be normalized across the elements in the batch
    DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION_NAME = 'softmax'
    DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION_EXTRA_ARGS = {'dim': -1}
    DEFAULT_LOSS_FUNCTION_NAME = 'crossentropyloss'
    DEFAULT_LOSS_FUNCTION_EXTRA_ARGS = {}
    DEFAULT_ACCURACY_FUNCTION_NAME = 'accuracy'
    DEFAULT_ACCURACY_FUNCTION_EXTRA_ARGS = {}

    def __init__(
            self,
            model: torch.nn.Module,
            # this class should accept params and lr
            optimizer_class_name: str = DEFAULT_OPTIMIZER_CLASS_NAME,
            # extra arguments for the optimizer
            optimizer_extra_args: typing.Optional[
                    typing.Dict[str, typing.Any]
            ] = DEFAULT_OPTIMIZER_EXTRA_ARGS,
            lr: float = DEFAULT_LEARNING_RATE,
            *,
            normalize_prob_func_name:
            str = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION_NAME,
            # extra arguments for the normalization function
            normalize_prob_func_extra_args: typing.Optional[
                    typing.Dict[str, typing.Any]
            ] = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION_EXTRA_ARGS,
            loss_func_name:
            str = DEFAULT_LOSS_FUNCTION_NAME,
            # extra arguments for the loss function
            loss_func_extra_args: typing.Optional[
                    typing.Dict[str, typing.Any]
            ] = DEFAULT_LOSS_FUNCTION_EXTRA_ARGS,
            accuracy_func_name:
            str = DEFAULT_ACCURACY_FUNCTION_NAME,
            # extra arguments for the accuracy function
            accuracy_func_extra_args: typing.Optional[
                    typing.Dict[str, typing.Any]
            ] = DEFAULT_ACCURACY_FUNCTION_EXTRA_ARGS,
    ):
        super().__init__()

        self.model = model
        # if we get a KeyError for the optimizer name, we raise a ValueError
        # specifying the list of supported optimizers from the dict
        try:
            self.optimizer_class = self.OPTIMIZERS_DICT[
                    optimizer_class_name.lower()
            ]
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
        # as before, if the name is not there we raise an error showing the
        # supported names
        # if found we instantiate it with the extra arguments
        try:
            self.normalize_prob_func = self.NORMALIZATIONS_DICT[
                    normalize_prob_func_name.lower()
            ](
                **normalize_prob_func_extra_args
            )
        except KeyError:
            raise ValueError(
                    'Please use one of the supported '
                    'normalizations: {}'.format(
                            tuple(self.NORMALIZATIONS_DICT.keys())
                    )
            )
        # as before, if the name is not there we raise an error showing the
        # supported names
        # if found we instantiate it with the extra arguments
        try:
            self.loss_func = self.LOSSES_DICT[
                    loss_func_name.lower()
            ](
                **loss_func_extra_args
            )
        except KeyError:
            raise ValueError(
                    'Please use one of the supported '
                    'losses: {}'.format(
                            tuple(self.LOSSES_DICT.keys())
                    )
            )
        # as before, if the name is not there we raise an error showing the
        # supported names
        # if found we instantiate it with the extra arguments
        try:
            self.accuracy_func = self.ACCURACIES_DICT[
                    accuracy_func_name.lower()
            ](
                **accuracy_func_extra_args
            )
        except KeyError:
            raise ValueError(
                    'Please use one of the supported '
                    'accuracies: {}'.format(
                            tuple(self.ACCURACIES_DICT.keys())
                    )
            )

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
