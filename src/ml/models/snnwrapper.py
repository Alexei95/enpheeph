import typing

import norse
import torch


class SNNReturnTuple(typing.NamedTuple):
    output: torch.Tensor
    state: torch.Tensor


class SNNWrapper(torch.nn.Module):
    DEFAULT_RETURN_STATE = True

    def __init__(
            self,
            encoder: typing.Callable[[torch.Tensor], torch.Tensor],
            model: torch.nn.Module,
            decoder: typing.Callable[[torch.Tensor], torch.Tensor],
            *,
            return_state: bool = DEFAULT_RETURN_STATE
    ):
        super().__init__()

        self.encoder = encoder
        self.model = model
        self.decoder = decoder

        self.return_state = return_state

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
        # we encode the inputs
        encoded_inputs = self.encoder(inputs)
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
                old_state = states[ts]
            else:
                old_state = state
            output, new_state = self.model(encoded_inputs[ts], old_state)
            # we append the output at the current timestep to the output list
            out.append(output)
            # also here we save the state in a list for returning it, otherwise
            # we save it just for the following execution
            if self.return_state:
                states[ts + 1] = new_state
            else:
                state = new_state

        # we stack the output to a torch tensor
        torch_out = torch.stack(out)
        # we decode the outputs
        decoded_output = self.decoder(torch_out)

        if self.return_state:
            return SNNReturnTuple(output=decoded_output, state=states)
        else:
            return decoded_output

    # NOTE: this is a temporary solution, as it is difficult to implement
    # temporary function with JSON
    @staticmethod
    def max_membrane_voltage_log_softmax_decoder(inputs):
        # we get the maximum for each membrane voltage over the time steps,
        # dim=0
        max_inputs, _ = torch.max(inputs, dim=0)
        outputs = torch.nn.functional.log_softmax(max_inputs, dim=-1)
        return outputs
