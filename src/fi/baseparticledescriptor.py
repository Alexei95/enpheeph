import dataclasses
import enum
import typing


# this is a container for the module name and the index for where the fault
# should be injected
# each fault descriptor covers a single bit-flip (or stuck-at)
@dataclasses.dataclass(init=True)
class BaseParticleDescriptor(object):
    # position to hit
    # we use
    pos_x: int
