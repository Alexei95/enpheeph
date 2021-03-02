# here we implement the model for the fault propagation in the hardware
# the basic idea is to partition the whole hardware chip (GPU, FPGA, ...)
# into small squares
# this matrix (which will be 3D in the future, to allow modeling also the third
# dimension when particles hit the chip) will have each square being associated
# at different levels with the most common element in there, i.e. for phyisical
# level it will be compute/memory/interconnection/...
# this will work at different layers (physical, architectural, software)
# for mapping the faults at multiple levels

# for post-poned annotations, required only from Python 3.7 to 3.9
from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import functools
import itertools
import typing

import networkx

from . import basefaultdescriptor

# make a single enum, using Physical_Compute instead of Compute and so on
# in this way we still use combinations Physical_Compute | Architectural_Register
# but we have unique identifiers for dicts and stuff, otherwise we would have
# collisions when numbering different enums
# NOTE: use enum._decompose(Enum, value) to get a list of the composing consts
CellType = enum.IntFlag('CellType',
                        [

                         # physical types
                         'Physical_Compute',
                         'Physical_Control',
                         'Physical_StaticMemory',
                         'Physical_DynamicMemory',
                         'Physical_Interconnection',

                         # architectural types
                         'Architectural_',

                         # software types
                         'Software_',

                        ],
                        module=__name__)




@dataclasses.dataclass(init=True)
class Cell(object):
    types: CellType
    _id_generator = dataclasses.field(init=False, repr=False,
                                      default=itertools.counter())
    _id: int = dataclasses.field(init=False)

    def __post_init__(self):
        self._id = next(self._id_generator)


@dataclasses.dataclass(init=True)
class HardwareModel(object):
    chip_width: float
    chip_height: float
    chip_measurement_unit: str = "mm^2"

    # this is a dict mapping each id to its own cell
    cells: typing.Dict[int, Cell]
    # this cells map is used to determine the hitting point of the fault
    # we use cell ids as elements, so that they can be used to get the cells
    # from the hash map
    cells_map: typing.Tuple[typing.Tuple[Cell, ...], ...]
    # we use a graph to represent the dependencies for computing the fault
    # propagation
    # the graph is compiled outside the model, and it is used for propagating
    # the effects of a particle strike in a predetermined location
    # cells_graph.add_edge(v1, v2, dependency_matrix={celltype1: prob1})
    # v1, v2 are the ids of the cells
    cells_graph: networkx.DiGraph

    def __post_init__(self):
        pass

    def inject_particle(self, particle):
