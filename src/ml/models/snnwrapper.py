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
        # we iterate over the timesteps
        for ts in range(seq_length):
            # we load the correct state depending on whether we are saving
            # them all or we only need it for execution
            if self.return_state:
                old_state = states[ts]
            else:
                old_state = state

            output, new_state = self.model(encoded_inputs[ts], old_state)
            # also here we save the state in a list for returning it, otherwise
            # we save it just for the following execution
            if self.return_state:
                states[ts + 1] = new_state
            else:
                state = new_state

        # we decode the outputs
        decoded_output = self.decoder(output)

        if self.return_state:
            return SNNReturnTuple(output=decoded_output, state=states)
        else:
            return decoded_output
