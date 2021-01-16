# this model class serves only as an interface for detailed implementation
# in our case, and it is built around a base model
# it uses dataclasses module from Python to ease further development
import dataclasses

import torch


@dataclasses.dataclass(init=True)
class Model:
    model: torch.nn.Module

    def __post_init__(self):
        pass
