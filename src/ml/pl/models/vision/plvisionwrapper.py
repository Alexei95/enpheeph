import collections
import typing

import pytorch_lightning
import torch
import torchmetrics


class PLVisionWrapper(
        pytorch_lightning.LightningModule,
):
    SCHEDULER_KEY = "scheduler"
    DEFAULT_INIT_BEFORE_FIT = True
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_BATCH_SIZE = 1
    DEFAULT_EXAMPLE_INPUT_ARRAY_SIZE = None
    DEFAULT_OPTIMIZER_CLASSES = [torch.optim.Adam]
    DEFAULT_LR_SCHEDULER_CLASSES = []
    # the default normalization function is softmax, and we compute it along
    # the last dimension as the first dimension is the batches, and we want
    # the results to be normalized across the elements in the batch
    DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION = torch.nn.Softmax(dim=-1)
    DEFAULT_LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    DEFAULT_PRE_ACCURACY_FUNCTION = torch.nn.Identity()
    DEFAULT_ACCURACY_FUNCTION = torchmetrics.Accuracy()

    def __init__(
            self,
            model: torch.nn.Module,
            init_before_fit: bool = DEFAULT_INIT_BEFORE_FIT,
            learning_rate: typing.Union[
                    float,
                    typing.Sequence[float]
            ] = DEFAULT_LEARNING_RATE,
            batch_size: int = DEFAULT_BATCH_SIZE,
            *,
            # used for exporting the model and producing summaries
            example_input_array_size:
            torch.Size = DEFAULT_EXAMPLE_INPUT_ARRAY_SIZE,
            # each class in this list should accept params and lr
            optimizer_classes: typing.Sequence[
                    typing.Callable[
                            [
                                    typing.Iterable[
                                            torch.nn.parameter.Parameter
                                    ],
                                    float
                            ],
                            torch.optim.Optimizer,
                    ]
            ] = DEFAULT_OPTIMIZER_CLASSES,
            # the schedules should also be a list of dicts with configurations
            # the classes in scheduler will be mapped 1-to-1 onto the optimizer
            # classes
            # hence, they should accept a singple argument which is the
            # corresponding optimizer
            lr_scheduler_classes: typing.Sequence[
                    typing.Dict[str, typing.Any]
            ] = DEFAULT_LR_SCHEDULER_CLASSES,
            normalize_prob_func: typing.Callable[
                        [torch.Tensor],
                        torch.Tensor,
            ] = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION,
            loss_func: typing.Callable[
                        [torch.Tensor, torch.Tensor],
                        torch.Tensor,
            ] = DEFAULT_LOSS_FUNCTION,
            pre_accuracy_func: typing.Callable[
                        [torch.Tensor],
                        torch.Tensor,
            ] = DEFAULT_PRE_ACCURACY_FUNCTION,
            accuracy_func: typing.Callable[
                        [torch.Tensor, torch.Tensor],
                        torch.Tensor,
            ] = DEFAULT_ACCURACY_FUNCTION,
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

        # we keep lr in the model to allow for Trainer.tune
        # to run and determine the optimal ones
        # we need to keep learning_rate as is to allow for tune to work
        self.learning_rate = self.hparams.learning_rate
        # same for batch size
        self.batch_size = self.hparams.batch_size

        self.normalize_prob_func = self.hparams.normalize_prob_func
        self.loss_func = self.hparams.loss_func
        self.pre_accuracy_func = self.hparams.pre_accuracy_func
        self.accuracy_func = self.hparams.accuracy_func

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

        error = "Schedulers should be at most as many as the optimizers"
        flag = len(self.optimizer_classes) >= len(self.lr_scheduler_classes)
        assert flag, error

        error = (
                "Each scheduler config dict must "
                "have a '{}' key".format(self.SCHEDULER_KEY)
        )
        assert all(
                self.SCHEDULER_KEY in d
                for d in self.lr_scheduler_classes
        ), error

    def configure_optimizers(self):
        self._check_lr_opt_sched()

        if isinstance(self.learning_rate, float):
            learning_rates = [
                    self.learning_rate
                    for _ in range(len(self.optimizer_classes))
            ]
        else:
            learning_rates = self.learning_rate
        optimizers = [
                opt(self.parameters(), lr=lr)
                for opt, lr in zip(self.optimizer_classes, learning_rates)
        ]
        lr_scheds = [
                # in this way we can save all the configurations
                # while overwriting the class with the correct object
                # instantiated using the corresponding optimizer
                {**d, self.SCHEDULER_KEY: d[self.SCHEDULER_KEY](opt)}
                # we can use this zip here as schedulers may be fewer
                # than optimzers, and we are interested only in schedulers
                for d, opt in zip(
                        self.lr_scheduler_classes,
                        optimizers
                )
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
