# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import collections
import functools
import typing

import norse
import pytorch_lightning
import pytorch_lightning.utilities.cli
import torch


class SNNReturnTuple(typing.NamedTuple):
    output: torch.Tensor
    state: torch.Tensor


# decorator to be used for running the proper loop with a forward of the main
# model
def snn_module_forward_decorator(model_forward):
    @functools.wraps(model_forward)
    def inner_forward(
        self,
        inputs: torch.Tensor,
        *,
        state: typing.Optional[typing.Sequence[typing.Tuple[torch.Tensor]]] = None,
    ) -> typing.Union[torch.Tensor, SNNReturnTuple]:

        # we encode the inputs, if enabled
        if self.encoding_flag:
            encoded_inputs = self.encoder(inputs)
        else:
            encoded_inputs = inputs
        # we save the sequence length from the shape of the inputs
        seq_length = encoded_inputs.size()[0]
        # states will contain the states at each time step, and the second
        # dimension will be the one covering the number of stateful layers
        # which returns states, which are named tuple
        # we initialize the states with the given ones, and then we add
        # new ones for covering the evolution of the system
        # this is done only if we will return the state at the end
        if self.return_state:
            states = [state] + [None] * seq_length

        # we need a list to save the output at each time step
        out = []
        # we iterate over the timesteps
        for ts in range(seq_length):
            # we load the correct state depending on whether we are saving
            # them all or we only need it for execution
            if self.return_state:
                state = states[ts]
            # we need to use self explicitly as this function is not
            # bound to an instance since it's wrapped
            output, state = model_forward(self, encoded_inputs[ts], state=state)
            # we append the output at the current timestep to
            # the output list
            out.append(output)
            # also here we save the state in a list for returning it,
            # otherwise we save it just for the following execution
            if self.return_state:
                states[ts + 1] = state

        # we stack the output to a torch tensor
        torch_out = torch.stack(out)
        # we decode the outputs, if enabled
        if self.decoding_flag:
            decoded_output = self.decoder(torch_out)
        else:
            decoded_output = output

        if self.return_state:
            return SNNReturnTuple(output=decoded_output, state=states)
        else:
            return decoded_output

    return inner_forward


class SNNModule(pytorch_lightning.LightningModule):
    DEFAULT_ENCODER = torch.nn.Identity()
    DEFAULT_DECODER = torch.nn.Identity()

    DEFAULT_RETURN_STATE = False
    DEFAULT_ENCODING_FLAG = True
    DEFAULT_DECODING_FLAG = True
    DEFAULT_TRAINABLE_NEURON_PARAMETERS = True

    SCHEDULER_KEY = "scheduler"
    DEFAULT_OPTIMIZER_CLASSES = (
        {
            "class_path": "torch.optim.Adam",
            "init_args": {},
        },
    )
    DEFAULT_LR_SCHEDULER_CLASSES = tuple(tuple())
    DEFAULT_LR_SCHEDULER_CONFIGS = tuple(tuple())
    # the default normalization function is softmax, and we compute it along
    # the last dimension as the first dimension is the batches, and we want
    # the results to be normalized across the elements in the batch
    DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION = {
        "class_path": "torch.nn.Softmax",
        "init_args": {"dim": -1},
    }
    DEFAULT_LOSS_FUNCTION = {
        "class_path": "torch.nn.CrossEntropyLoss",
        "init_args": {},
    }
    DEFAULT_PRE_ACCURACY_FUNCTION = {
        "class_path": "torch.nn.Identity",
        "init_args": {},
    }
    DEFAULT_ACCURACY_FUNCTION = {
        "class_path": "torchmetrics.Accuracy",
        "init_args": {},
    }
    DEFAULT_DIMS = None
    DEFAULT_NUM_CLASSES = None
    DIMS = (1, 128, 128)
    NUM_CLASSES = 11

    def __init__(
        self,
        *args: typing.Any,
        encoder: typing.Callable[[torch.Tensor], torch.Tensor] = DEFAULT_ENCODER,
        decoder: typing.Callable[[torch.Tensor], torch.Tensor] = DEFAULT_DECODER,
        return_state: bool = DEFAULT_RETURN_STATE,
        encoding_flag: bool = DEFAULT_ENCODING_FLAG,
        decoding_flag: bool = DEFAULT_DECODING_FLAG,
        trainable_neuron_parameters: bool = DEFAULT_TRAINABLE_NEURON_PARAMETERS,
        # each class in this list should accept params and lr
        # this should be
        # typing.Sequence[typing.Callable[[typing.Iterable[
        # torch.nn.parameter.Parameter], float, typing.Any, ...],
        # torch.optim.Optimizer]
        # but Callable cannot be used for jsonargparse to work properly
        optimizer_classes: typing.Sequence[typing.Dict] = DEFAULT_OPTIMIZER_CLASSES,
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
        lr_scheduler_classes: typing.Sequence[
            typing.Sequence[typing.Dict]
        ] = DEFAULT_LR_SCHEDULER_CLASSES,
        # this is for configurations of the learning rate schedulers
        lr_scheduler_configs: typing.Sequence[
            typing.Sequence[typing.Dict]
        ] = DEFAULT_LR_SCHEDULER_CONFIGS,
        # all these ones should be
        # typing.Callable[[torch.Tensor], torch.Tensor]
        # but Callable cannot be used for jsonargparse to work properly
        normalize_prob_func: typing.Any = DEFAULT_PROBABILITY_NORMALIZATION_FUNCTION,
        loss_func: typing.Any = DEFAULT_LOSS_FUNCTION,
        pre_accuracy_func: typing.Any = DEFAULT_PRE_ACCURACY_FUNCTION,
        accuracy_func: typing.Any = DEFAULT_ACCURACY_FUNCTION,
        dims: typing.Optional[typing.Sequence[int]] = DEFAULT_DIMS,
        num_classes: typing.Optional[int] = DEFAULT_NUM_CLASSES,
        **kwargs: typing.Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.encoder = pytorch_lightning.utilities.cli.instantiate_class(
            tuple(), self.hparams.encoder
        )
        self.decoder = pytorch_lightning.utilities.cli.instantiate_class(
            tuple(), self.hparams.decoder
        )

        self.encoding_flag = self.hparams.encoding_flag
        self.decoding_flag = self.hparams.decoding_flag

        self.return_state = self.hparams.return_state

        self.trainable_neuron_parameters = self.hparams.trainable_neuron_parameters

        self.optimizer_classes = self.hparams.optimizer_classes
        self.lr_scheduler_classes = self.hparams.lr_scheduler_classes
        self.lr_scheduler_configs = self.hparams.lr_scheduler_configs

        self.normalize_prob_func = pytorch_lightning.utilities.cli.instantiate_class(
            tuple(), self.hparams.normalize_prob_func
        )
        self.loss_func = pytorch_lightning.utilities.cli.instantiate_class(
            tuple(), self.hparams.loss_func
        )
        self.pre_accuracy_func = pytorch_lightning.utilities.cli.instantiate_class(
            tuple(), self.hparams.pre_accuracy_func
        )
        self.accuracy_func = pytorch_lightning.utilities.cli.instantiate_class(
            tuple(), self.hparams.accuracy_func
        )
        # we save the input size
        self.dims = self.hparams.dims
        if self.dims is None and hasattr(self, "DIMS"):
            self.dims = self.DIMS
        # we save the number of classes
        self.num_classes = self.hparams.num_classes
        if self.num_classes is None and hasattr(self, "NUM_CLASSES"):
            self.num_classes = self.NUM_CLASSES

        self.lazy_model_init()
        self.post_init()

    def post_init(self):
        # where to place eventual fixes for LIF JIT functions

        self.register_snn_parameters()

    def check_hyperparameters(self):
        callable_ = (
            callable(self.normalize_prob_func)
            and callable(self.loss_func)
            and callable(self.pre_accuracy_func)
            and callable(self.accuracy_func)
        )
        if not callable_:
            raise ValueError("The functions should be callable")

        self._check_encoder_decoder()

        self._check_lr_opt_sched()

    def _check_encoder_decoder(self):
        callable_ = callable(self.encoder) and callable(self.decoder)
        if not callable_:
            raise ValueError("The encoder/decoder should be callable")

    # this method is used to register possible hidden parameters inside the
    # SNN configurations
    def register_snn_parameters(self):
        # we get all the Parameter elements from the modules
        # some Parameters have nested Parameters, like LIFRefrac has
        # a nested LIFParameters in it
        p_list = []
        # we need a counter as many parameters may have the same name
        counter = 0

        # we populate the list with direct children to the modules,
        # using 'p' as variable name
        # only if it is a namedtuple, with _asdict, or if it is a
        # torch.nn.Module
        for module in self.modules():
            if hasattr(module, "p"):
                p = getattr(module, "p")
                if hasattr(p, "_asdict"):
                    p_list.extend(list(p._asdict().items()))
                elif isinstance(p, torch.nn.Module):
                    p_list.extend(list(p.named_modules()))

        # we iterate over the list until it's empty
        while len(p_list) > 0:
            p_name, p_value = p_list.pop()

            # if the value is a namedtuple or a torch.nn.Module we extend the
            # list
            if hasattr(p_value, "_asdict"):
                p_list.extend(list(p_value._asdict().items()))
            elif isinstance(p_value, torch.nn.Module):
                p_list.extend(list(p_value.named_modules()))
            # we check wheter it is a tensor which requires gradient and
            # it is not already registered
            tensor_flag = isinstance(p_value, torch.Tensor)
            grad_flag = getattr(p_value, "requires_grad", False)
            id_param_list = [id(param) for param in self.parameters()]
            parameter_flag = id(p_value) not in id_param_list
            # if True we increase the counter and register the new parameter
            if tensor_flag and grad_flag and parameter_flag:
                counter += 1
                module.register_parameter("p/" + p_name + "/" + str(counter), p_value)

    # we delegate the weight initialization to each component
    # decoder, model, encoder
    def init_weights(self):
        for mod in (self.decoder, self.encoder):
            if (init_weights := getattr(mod, "init_weights", None)) is not None:
                init_weights()
        # this initialization is similar to the ResNet one
        # taken from https://github.com/Lornatang/AlexNet-PyTorch/
        # @ alexnet_pytorch/model.py#L63
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    # implemented by us for compatibility between forward and validation/test
    # steps
    def inference_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.normalize_prob_func(self.forward(x))
        loss = self.loss_func(y_hat, y)
        acc = self.accuracy_func(self.pre_accuracy_func(y_hat), y)

        return {"acc": acc, "loss": loss}

    def training_step(self, batch, batch_idx):
        m = self.inference_step(batch, batch_idx)
        metrics = {
            "train_acc": m["acc"],
            "train_loss": m["loss"],
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        # here we need to return the loss to be able to properly train
        return m["loss"]

    def validation_step(self, batch, batch_idx):
        m = self.inference_step(batch, batch_idx)
        metrics = {
            "val_acc": m["acc"],
            "val_loss": m["loss"],
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        # this may not be needed, as for logging we already use self.log_dict
        # return metrics

    def test_step(self, batch, batch_idx):
        m = self.inference_step(batch, batch_idx)
        metrics = {
            "test_acc": m["acc"],
            "test_loss": m["loss"],
        }
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        # this may not be needed, as for logging we already use self.log_dict
        # return metrics

    # this function is used to check that lr, optimizers and schedulers
    # follow the general rules
    # lr can be either one for all optimizers or one for each
    # schedulers can be in any number up to optimizers and they will
    # be fed the corresponding optimizer in the list
    def _check_lr_opt_sched(self):
        assert isinstance(
            self.optimizer_classes, collections.abc.Sequence
        ), "Optimizer classes must be a list"
        assert isinstance(
            self.learning_rate, (float, collections.abc.Sequence)
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
                self.learning_rate for _ in range(len(self.optimizer_classes))
            )
            self.learning_rate = learning_rates

        error = (
            "List of scheduler lists and config lists should be "
            "the same length as the optimizer list"
        )
        flag = len(self.optimizer_classes) == len(self.lr_scheduler_classes)
        flag2 = len(self.optimizer_classes) == len(self.lr_scheduler_configs)
        assert flag and flag2, error

    def configure_optimizers(self):
        self._check_lr_opt_sched()

        optimizers = [
            pytorch_lightning.utilities.cli.instantiate_class(
                (self.parameters(), lr), init=opt
            )
            for opt, lr in zip(self.optimizer_classes, self.learning_rate)
        ]
        lr_scheds = [
            # in this way we can save all the configurations
            # while overwriting the class with the correct object
            # instantiated using the corresponding optimizer
            {
                **config,
                self.SCHEDULER_KEY: (
                    pytorch_lightning.utilities.cli.instantiate_class(opt, class_)
                ),
            }
            # we zip over the lists for scheduler classes and configs
            for sublist_classes, sublist_configs, opt in zip(
                self.lr_scheduler_classes, self.lr_scheduler_configs, optimizers
            )
            # we go over each scheduler and its config in the sublists
            for class_, config in zip(sublist_classes, sublist_configs)
        ]
        return optimizers, lr_scheds

    def lazy_model_init(self):
        if self.trainable_neuron_parameters:
            lif1 = norse.torch.LIFCell(
                p=norse.torch.LIFParameters(
                    tau_syn_inv=torch.nn.Parameter(
                        torch.full(
                            size=[32, 32, 32],
                            fill_value=(
                                norse.torch.LIFParameters._field_defaults.get(
                                    "tau_syn_inv"
                                )
                            ),
                        ),
                    ),
                    tau_mem_inv=torch.nn.Parameter(
                        torch.full(
                            size=[32, 32, 32],
                            fill_value=(
                                norse.torch.LIFParameters._field_defaults.get(
                                    "tau_mem_inv"
                                )
                            ),
                        ),
                    ),
                    v_leak=torch.nn.Parameter(
                        norse.torch.LIFParameters._field_defaults.get("v_leak")
                    ),
                    v_th=torch.nn.Parameter(
                        torch.full(
                            size=[32, 32, 32],
                            fill_value=(
                                0.4
                                # norse.torch.LIFParameters.
                                # _field_defaults.get(
                                #         "v_th"
                                # )
                            ),
                        ),
                    ),
                    v_reset=torch.nn.Parameter(
                        torch.full(
                            size=[32, 32, 32],
                            fill_value=(
                                norse.torch.LIFParameters._field_defaults.get("v_reset")
                            ),
                        ),
                    ),
                    alpha=norse.torch.LIFParameters._field_defaults.get("alpha"),
                    method="super",
                ),
                dt=0.01,
            )
            lif2 = norse.torch.LIFCell(
                p=norse.torch.LIFParameters(
                    tau_syn_inv=torch.nn.Parameter(
                        torch.full(
                            size=[32, 16, 16],
                            fill_value=(
                                norse.torch.LIFParameters._field_defaults.get(
                                    "tau_syn_inv"
                                )
                            ),
                        ),
                    ),
                    tau_mem_inv=torch.nn.Parameter(
                        torch.full(
                            size=[32, 16, 16],
                            fill_value=(
                                norse.torch.LIFParameters._field_defaults.get(
                                    "tau_mem_inv"
                                )
                            ),
                        ),
                    ),
                    v_leak=torch.nn.Parameter(
                        norse.torch.LIFParameters._field_defaults.get("v_leak")
                    ),
                    v_th=torch.nn.Parameter(
                        torch.full(
                            size=[32, 16, 16],
                            fill_value=(
                                0.4
                                # norse.torch.LIFParameters.
                                # _field_defaults.get(
                                #         "v_th"
                                # )
                            ),
                        ),
                    ),
                    v_reset=torch.nn.Parameter(
                        torch.full(
                            size=[32, 16, 16],
                            fill_value=(
                                norse.torch.LIFParameters._field_defaults.get("v_reset")
                            ),
                        ),
                    ),
                    alpha=norse.torch.LIFParameters._field_defaults.get("alpha"),
                    method="super",
                ),
                dt=0.01,
            )
            li = norse.torch.LICell(
                p=norse.torch.LIParameters(
                    tau_syn_inv=torch.nn.Parameter(
                        torch.full(
                            size=[11],
                            fill_value=(
                                norse.torch.LIParameters._field_defaults.get(
                                    "tau_syn_inv"
                                )
                            ),
                        ),
                    ),
                    tau_mem_inv=torch.nn.Parameter(
                        torch.full(
                            size=[11],
                            fill_value=(
                                norse.torch.LIParameters._field_defaults.get(
                                    "tau_mem_inv"
                                )
                            ),
                        ),
                    ),
                    v_leak=torch.nn.Parameter(
                        norse.torch.LIParameters._field_defaults.get("v_leak")
                    ),
                ),
                dt=torch.nn.Parameter(
                    torch.full(
                        size=[11],
                        fill_value=0.01,
                    ),
                ),
            )
        else:
            lif1 = norse.torch.LIFCell()
            lif2 = norse.torch.LIFCell()
            li = norse.torch.LICell()

        self.sequential = norse.torch.SequentialState(
            torch.nn.AvgPool2d(
                kernel_size=4,
                stride=4,
                padding=0,
                ceil_mode=False,
            ),
            torch.nn.Dropout(
                p=0.1,
                inplace=False,
            ),
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                dilation=1,
                stride=1,
                groups=1,
            ),
            lif1,
            torch.nn.AvgPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
                ceil_mode=False,
            ),
            torch.nn.Dropout(
                p=0.1,
                inplace=False,
            ),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
                dilation=1,
                stride=1,
                groups=1,
            ),
            lif2,
            torch.nn.AvgPool2d(
                kernel_size=2,
                stride=2,
                padding=0,
                ceil_mode=False,
            ),
            torch.nn.Dropout(
                p=0.2,
                inplace=False,
            ),
            torch.nn.Flatten(
                start_dim=1,
                end_dim=-1,
            ),
            torch.nn.Linear(
                in_features=2048,
                out_features=500,
                bias=True,
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=500,
                out_features=11,
                bias=True,
            ),
            li,
        )

        # this must be called after setting up the SNN module
        self.register_snn_parameters()

    @snn_module_forward_decorator
    def forward(self, x, state=None):
        return self.sequential.forward(x, state=state)
