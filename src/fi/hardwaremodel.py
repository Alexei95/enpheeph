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
import typing

from . import basefaultdescriptor

# make a single enum, using Physical_Compute instead of Compute and so on
# in this way we still use combinations Physical_Compute | Architectural_Register
# but we have unique identifiers for dicts and stuff, otherwise we would have
# collisions when numbering different enums
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
    types: typing.List[enum.IntFlag, ...] = dataclasses.field(init=True,
                                                        default_factory=list)
    _tree_default_dict_factory = dataclasses.field(init=False,
                    default=functools.partial(collections.defaultdict, list))
    _dependency_default_dict_factory = dataclasses.field(init=False,
                    default=functools.partial(collections.defaultdict, 0))

    # these lists are used to have all the predecessors and successors for
    # propagating the fault effects
    # siblings is for cells at the same tree level
    predecessors: typing.DefaultDict[CellType, typing.List[Cell, ...]] = \
        dataclasses.field(init=True,
                          default_factory=_tree_default_dict_factory)
    siblings: typing.DefaultDict[CellType, typing.List[Cell, ...]] = \
        dataclasses.field(init=True,
                          default_factory=_tree_default_dict_factory)
    successors: typing.DefaultDict[CellType, typing.List[Cell, ...]] = \
        dataclasses.field(init=True,
                          default_factory=_tree_default_dict_factory)

    # these dicts associate each cell type to the propagation probability
    # generally we consider only successors, but in case also predecessors and
    # siblings can be considered
    # each of them contain the probability [0, 1] to propagate the fault
    dependency_predecessors: typing.DefaultDict[CellType, float] = \
        dataclasses.field(init=True,
                          default_factory=_dependency_default_dict_factory)
    dependency_siblings: typing.DefaultDict[CellType, float] = \
        dataclasses.field(init=True,
                          default_factory=_dependency_default_dict_factory)
    dependency_successors: typing.DefaultDict[CellType, float] = \
        dataclasses.field(init=True,
                          default_factory=_dependency_default_dict_factory)

    def __post_init__(self):
        pass

    def add_sibling(self, celltype: CellType, cell: Cell):
        self.siblings[celltype].append(cell)

    def add_predecessor(self, celltype: CellType, cell: Cell):
        self.predecessors[celltype].append(cell)

    def add_successor(self, celltype: CellType, cell: Cell):
        self.successors[celltype].append(cell)

    def reset_siblings(self, celltype: typing.Union[CellType, None] = None):
        if celltype is None:
            self.siblings = self._tree_default_dict_factory()
        del self.siblings[celltype]

    def reset_predecessors(self, celltype: typing.Union[CellType, None] = None):
        if celltype is None:
            self.predecessors = self._tree_default_dict_factory()
        del self.predecessors[celltype]

    def reset_successors(self, celltype: typing.Union[CellType, None] = None):
        if celltype is None:
            self.successors = self._tree_default_dict_factory()
        del self.successors[celltype]





@dataclasses.dataclass(init=True)
class HardwareModel(object):
    chip_width: float
    chip_height: float
    chip_measurement_unit: str = "mm^2"

    cells: typing.Tuple[typing.Tuple[Cell, ...], ...]

    def __post_init__(self):
        pass
