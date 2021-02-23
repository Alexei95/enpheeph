# here we generate the faults, starting from a probabilistic model
# we consider also the hardware model, covering the physical, architectural and
# software implementations, as well as taking into account properties from
# the model itself, such as channels, layer dimensions, etc.

import dataclasses

@dataclasses.dataclass(init=True)
class FaultGeneration(object):
    sampler
    hardware_model
