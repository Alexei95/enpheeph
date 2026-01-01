# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2026 Alessio "Alexei95" Colucci
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

import functools
import typing

import norse
import pytorch_lightning
import pytorch_lightning.utilities.cli
import torch
import torchmetrics
import torchvision


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


class DVS128GestureSNNModule(pytorch_lightning.LightningModule):
    DEFAULT_ENCODER = torch.nn.Identity()
    DEFAULT_DECODER = torch.nn.Identity()

    DEFAULT_OPTIMIZER_CLASS = torch.optim.Adam
    DEFAULT_LEARNING_RATE = 1e-3

    DEFAULT_RETURN_STATE = False
    DEFAULT_ENCODING_FLAG = True
    DEFAULT_DECODING_FLAG = True
    DEFAULT_TRAINABLE_NEURON_PARAMETERS = True

    DEFAULT_EXAMPLE_INPUT_ARRAY_SIZE = (1, 1, 1, 128, 128)
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
        dims: typing.Optional[typing.Sequence[int]] = DIMS,
        num_classes: typing.Optional[int] = NUM_CLASSES,
        example_input_array_size: typing.Optional[
            typing.Sequence[int]
        ] = DEFAULT_EXAMPLE_INPUT_ARRAY_SIZE,
        optimizer_class: type(torch.optim.Optimizer) = DEFAULT_OPTIMIZER_CLASS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        map: typing.Optional[torch.device] = None,
        **kwargs: typing.Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.encoder = encoder
        self.decoder = decoder

        self.encoding_flag = self.hparams.encoding_flag
        self.decoding_flag = self.hparams.decoding_flag

        self.return_state = self.hparams.return_state

        self.trainable_neuron_parameters = self.hparams.trainable_neuron_parameters

        self.optimizer_classes = optimizer_class
        self.learning_rates = learning_rate

        self.normalize_prob_func = torch.nn.Identity()
        self.pre_accuracy_func = torch.nn.Identity()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.accuracy_func = self.custom_argmax_accuracy

        # we save the input size
        self.dims = dims
        if self.dims is None and hasattr(self, "DIMS"):
            self.dims = self.DIMS
        # we save the number of classes
        self.num_classes = num_classes
        if self.num_classes is None and hasattr(self, "NUM_CLASSES"):
            self.num_classes = self.NUM_CLASSES
        self.example_input_array_size = example_input_array_size
        if self.example_input_array_size is not None:
            self.example_input_array = torch.randn(*self.example_input_array_size)

        self._check_encoder_decoder()

        self.model_definition()

        if map is not None:
            self.to(map)

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
                p = module.p
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

    def configure_optimizers(self):
        optimizer = self.optimizer_classes(self.parameters(), self.learning_rates)
        return optimizer

    def model_definition(self):
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
            # 2
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
            # 6
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
            # 11
            torch.nn.Linear(
                in_features=2048,
                out_features=500,
                bias=True,
            ),
            torch.nn.ReLU(),
            # 13
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

    # NOTE: this is a temporary solution, as it is difficult to implement
    # temporary function with JSON
    @staticmethod
    def random_noise_max_membrane_voltage_log_softmax_decoder(inputs):
        # we add some random noise
        temp = inputs + 0.001 * torch.randn(*inputs.size(), device=inputs.device)
        # we get the maximum for each membrane voltage over the time steps,
        # dim=0
        max_inputs, _ = torch.max(temp, dim=0)
        return max_inputs

    # NOTE: this is a temporary solution, as it is difficult to implement
    # temporary function with JSON
    @staticmethod
    def label_smoothing_loss(y_hat, y, alpha=0.2):
        log_probs = torch.nn.functional.log_softmax(y_hat, dim=-1)
        xent = torch.nn.functional.nll_loss(log_probs, y, reduction="none")
        KL = -log_probs.mean(dim=-1)
        loss = (1 - alpha) * xent + alpha * KL
        return loss.sum()

    @staticmethod
    def custom_softmax_accuracy(y_hat, y):
        return torchmetrics.Accuracy().to(y_hat.device)(
            torch.nn.functional.softmax(y_hat, dim=-1), y
        )

    # the following functions are for MNIST SNN training, from the norse
    # tutorial
    @staticmethod
    def custom_argmax_accuracy(y_hat, y):
        return torchmetrics.Accuracy().to(y_hat.device)(torch.argmax(y_hat, dim=-1), y)

    # must be used if the target is one-hot encoded
    @staticmethod
    def custom_one_hot_argmax_accuracy(y_hat, y):
        return torchmetrics.Accuracy().to(y_hat.device)(
            torch.argmax(y_hat, dim=-1),
            torch.max(y, dim=-1)[1],
        )

    @staticmethod
    def max_log_softmax_probability(x):
        x, _ = torch.max(x, 0)
        log_p_y = torch.nn.functional.log_softmax(x, dim=-1)
        return log_p_y

    @staticmethod
    def decoder_dvs128gesture(x):
        return DVS128GestureSNNModule.max_log_softmax_probability(x)

    @classmethod
    def encoder_dvs128gesture(cls, input_):
        encoder_name = "_encoder_dvs128gesture"
        if (encoder := getattr(cls, encoder_name, None)) is None:
            encoder = torchvision.transforms.Compose(
                [
                    lambda x: x.to_dense() if x.is_sparse else x,
                    lambda x: x[:, :, 0:1, :, :],
                    functools.partial(
                        lambda x, dtype: x.to(dtype=dtype) if x.dtype != dtype else x,
                        dtype=torch.float32,
                    ),
                    lambda x: x.permute(1, 0, 2, 3, 4),
                ]
            )
            setattr(cls, encoder_name, encoder)
        return encoder(input_)
