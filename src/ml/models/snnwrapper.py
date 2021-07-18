import itertools
import typing

import norse
import torch
import torchmetrics


class SNNReturnTuple(typing.NamedTuple):
    output: torch.Tensor
    state: torch.Tensor


class SNNWrapper(torch.nn.Module):
    DEFAULT_RETURN_STATE = True
    DEFAULT_ENCODING_FLAG = True
    DEFAULT_DECODING_FLAG = True

    def __init__(
            self,
            encoder: typing.Callable[[torch.Tensor], torch.Tensor],
            model: torch.nn.Module,
            decoder: typing.Callable[[torch.Tensor], torch.Tensor],
            *,
            return_state: bool = DEFAULT_RETURN_STATE,
            encoding_flag: bool = DEFAULT_ENCODING_FLAG,
            decoding_flag: bool = DEFAULT_DECODING_FLAG,
    ):
        super().__init__()

        self.encoder = encoder
        self.model = model
        self.decoder = decoder

        self.encoding_flag = encoding_flag
        self.decoding_flag = decoding_flag

        self.register_snn_parameters()

        self.return_state = return_state

    # this method is used to register possible hidden parameters inside the
    # SNN configurations
    def register_snn_parameters(self):
        for module_name, module in self.named_modules():
            if hasattr(module, 'p'):
                p = getattr(module, 'p')
                if hasattr(p, '_asdict'):
                    for p_name, p_value in p._asdict().items():
                        if getattr(p_value, 'requires_grad', False):
                            module.register_parameter(
                                'p_' + p_name,
                                p_value
                            )

    def forward(
            self,
            inputs: torch.Tensor,
            *,
            state: typing.Optional[
                    typing.Sequence[
                            torch.Tensor
                    ]
            ] = None,
    ) -> typing.Union[
            torch.Tensor,
            SNNReturnTuple
    ]:
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
            output, state = self.model(encoded_inputs[ts], state)
            # we append the output at the current timestep to the output list
            out.append(output)
            # also here we save the state in a list for returning it, otherwise
            # we save it just for the following execution
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

    # NOTE: this is a temporary solution, as it is difficult to implement
    # temporary function with JSON
    @staticmethod
    def random_noise_max_membrane_voltage_log_softmax_decoder(inputs):
        # we add some random noise
        temp = inputs + 0.001 * torch.randn(
                *inputs.size(), device=inputs.device
        )
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
        KL = - log_probs.mean(dim=-1)
        loss = (1 - alpha) * xent + alpha * KL
        return loss.sum()

    @staticmethod
    def custom_softmax_accuracy(y_hat, y):
        return torchmetrics.Accuracy().to(y_hat.device)(
                torch.nn.functional.softmax(y_hat, dim=-1),
                y
        )

    # the following functions are for MNIST SNN training, from the norse
    # tutorial
    @staticmethod
    def custom_argmax_accuracy(y_hat, y):
        return torchmetrics.Accuracy().to(y_hat.device)(
                torch.argmax(y_hat, dim=-1),
                y
        )

    @staticmethod
    def max_log_softmax_proability(x):
        x, _ = torch.max(x, 0)
        log_p_y = torch.nn.functional.log_softmax(x, dim=-1)
        return log_p_y
