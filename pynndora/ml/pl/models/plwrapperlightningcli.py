import collections
import typing

import pytorch_lightning
import pytorch_lightning.utilities.cli
import torch
import torchmetrics


class PLWrapperLightningCLI(
        pytorch_lightning.LightningModule,
):
    SCHEDULER_KEY = "scheduler"
    DEFAULT_INIT_BEFORE_FIT = True
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_EXAMPLE_INPUT_ARRAY_SIZE = None
    DEFAULT_OPTIMIZER_CLASSES = (
            {
                    'class_path': 'torch.optim.Adam',
                    'init_args': {},
            },
    )
    DEFAULT_LR_SCHEDULER_CLASSES = tuple(tuple())
    DEFAULT_LR_SCHEDULER_CONFIGS = tuple(tuple())
    # the default normalization function is softmax, and we compute it along
    # the last dimension as the first dimension is the batches, and we want
    # the results to be normalized across the elements in the batch
    DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION = {
            'class_path': 'torch.nn.Softmax',
            'init_args': {
                    'dim': -1
            },
    }
    DEFAULT_LOSS_FUNCTION = {
            'class_path': 'torch.nn.CrossEntropyLoss',
            'init_args': {},
    }
    DEFAULT_PRE_ACCURACY_FUNCTION = {
            'class_path': 'torch.nn.Identity',
            'init_args': {},
    }
    DEFAULT_ACCURACY_FUNCTION = {
            'class_path': 'torchmetrics.Accuracy',
            'init_args': {},
    }

    def __init__(
            self,
            *,
            model: torch.nn.Module,
            init_before_fit: bool = DEFAULT_INIT_BEFORE_FIT,
            learning_rate: typing.Union[
                    float,
                    typing.Sequence[float]
            ] = DEFAULT_LEARNING_RATE,
            batch_size: int = DEFAULT_BATCH_SIZE,
            # used for exporting the model and producing summaries
            example_input_array_size:
            typing.Optional[torch.Size] = DEFAULT_EXAMPLE_INPUT_ARRAY_SIZE,
            # each class in this list should accept params and lr
            # this should be
            # typing.Sequence[typing.Callable[[typing.Iterable[
            # torch.nn.parameter.Parameter], float, typing.Any, ...],
            # torch.optim.Optimizer]
            # but Callable cannot be used for jsonargparse to work properly
            optimizer_classes: typing.Sequence[
                    typing.Dict
            ] = DEFAULT_OPTIMIZER_CLASSES,
            # the schedules should also be a list of dicts with configurations
            # the classes in scheduler will be mapped 1-to-1 onto the optimizer
            # classes
            # hence, they should accept a singple argument which is the
            # corresponding optimizer
            # this should be
            # typing.Sequence[typing.Callable[[torch.optim.Optimizer,
            # typing.Any, ...],
            # torch.optim.lr_scheduler._LRScheduler]
            # but Callable cannot be used for jsonargparse to work properly
            lr_scheduler_classes: typing.Sequence[typing.Sequence[
                    typing.Dict
            ]] = DEFAULT_LR_SCHEDULER_CLASSES,
            # this is for configurations of the learning rate schedulers
            lr_scheduler_configs: typing.Sequence[typing.Sequence[
                    typing.Dict
            ]] = DEFAULT_LR_SCHEDULER_CONFIGS,
            # all these ones should be
            # typing.Callable[[torch.Tensor], torch.Tensor]
            # but Callable cannot be used for jsonargparse to work properly
            normalize_prob_func:
            typing.Any = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION,
            loss_func:
            typing.Any = DEFAULT_LOSS_FUNCTION,
            pre_accuracy_func:
            typing.Any = DEFAULT_PRE_ACCURACY_FUNCTION,
            accuracy_func:
            typing.Any = DEFAULT_ACCURACY_FUNCTION,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = self.hparams.model
        # we save the flag to actually trigger the init when the warm-up
        # function is called
        self.init_before_fit = self.hparams.init_before_fit
        # we generate the random tensor from the size used as input
        self.example_input_array_size = self.hparams.example_input_array_size
        if self.example_input_array_size is not None:
            self.example_input_array = torch.randn(
                    *self.example_input_array_size
            )
        self.optimizer_classes = self.hparams.optimizer_classes
        self.lr_scheduler_classes = self.hparams.lr_scheduler_classes
        self.lr_scheduler_configs = self.hparams.lr_scheduler_configs

        # we keep lr in the model to allow for Trainer.tune
        # to run and determine the optimal ones
        # we need to keep learning_rate as is to allow for tune to work
        self.learning_rate = self.hparams.learning_rate
        # same for batch size
        self.batch_size = self.hparams.batch_size

        self.normalize_prob_func = (
                pytorch_lightning.utilities.cli.
                instantiate_class(tuple(), self.hparams.normalize_prob_func)
        )
        self.loss_func = (
                pytorch_lightning.utilities.cli.
                instantiate_class(tuple(), self.hparams.loss_func)
        )
        self.pre_accuracy_func = (
                pytorch_lightning.utilities.cli.
                instantiate_class(tuple(), self.hparams.pre_accuracy_func)
        )
        self.accuracy_func = (
                pytorch_lightning.utilities.cli.
                instantiate_class(tuple(), self.hparams.accuracy_func)
        )

        self._check_hyperparameters()

    def _check_hyperparameters(self):
        callable_ = (
                callable(self.normalize_prob_func) and
                callable(self.loss_func) and
                callable(self.pre_accuracy_func) and
                callable(self.accuracy_func)
        )
        if not callable_:
            raise ValueError("The functions should be callable")

        self._check_lr_opt_sched()

    def forward(self, input_):
        return self.model(input_)

    # implemented by us for compatibility between forward and validation/test
    # steps
    def inference_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.normalize_prob_func(self.forward(x))
        loss = self.loss_func(y_hat, y)
        acc = self.accuracy_func(self.pre_accuracy_func(y_hat), y)

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
                logger=True
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
                on_epoch=True,
                logger=True
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
                on_epoch=True,
                logger=True
        )
        # this may not be needed, as for logging we already use self.log_dict
        # return metrics

    # this function is used to check that lr, optimizers and schedulers
    # follow the general rules
    # lr can be either one for all optimizers or one for each
    # schedulers can be in any number up to optimizers and they will
    # be fed the corresponding optimizer in the list
    def _check_lr_opt_sched(self):
        assert isinstance(
                self.optimizer_classes,
                collections.abc.Sequence
        ), "Optimizer classes must be a list"
        assert isinstance(
                self.learning_rate,
                (float, collections.abc.Sequence)
        ), "LR must be either a float or a list"

        if isinstance(self.learning_rate, collections.abc.Sequence):
            error = (
                    "Learning rates in a list must be provided "
                    "for all optimzers one-to-one"
            )
            flag = len(self.learning_rate) == len(self.optimizer_classes)
            assert flag, error

        if isinstance(self.learning_rate, float):
            learning_rates = tuple(
                    self.learning_rate
                    for _ in range(len(self.optimizer_classes))
            )
            self.learning_rate = learning_rates

        error = (
                "List of scheduler lists and config lists should be "
                "the same length as the optimizer list"
        )
        flag = len(self.optimizer_classes) == len(self.lr_scheduler_classes)
        flag2 = len(self.optimizer_classes) == len(self.lr_scheduler_configs)
        assert (flag and flag2), error

    def configure_optimizers(self):
        optimizers = [
                pytorch_lightning.utilities.cli.
                instantiate_class((self.parameters(), lr), init=opt)
                for opt, lr in zip(self.optimizer_classes, self.learning_rate)
        ]
        lr_scheds = [
                # in this way we can save all the configurations
                # while overwriting the class with the correct object
                # instantiated using the corresponding optimizer
                {
                        **config,
                        self.SCHEDULER_KEY: (
                                pytorch_lightning.utilities.cli.
                                instantiate_class(
                                        opt, class_
                                )
                        ),
                }
                # we zip over the lists for scheduler classes and configs
                for sublist_classes, sublist_configs, opt in zip(
                        self.lr_scheduler_classes,
                        self.lr_scheduler_configs,
                        optimizers
                )
                # we go over each scheduler and its config in the sublists
                for class_, config in zip(sublist_classes, sublist_configs)
        ]
        return optimizers, lr_scheds

    # this function is called at the beginning of the training, so it
    # can be used for weight initialization
    def on_fit_start(self):
        # if enabled
        if self.init_before_fit:
            # we try to init the weights, if it doesn't exist we skip
            if (init_weights := getattr(
                    self.model,
                    "init_weights",
                    None
            )) is not None:
                init_weights()
