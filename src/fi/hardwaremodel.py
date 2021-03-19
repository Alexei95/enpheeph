# for forward annotations, when using the defined class inside the class itself
# for annotations
# only required for Python 3.7 up to 3.9, standard from 3.10
# for Python 3.6 and earlier, use a string with the name of the class
# https://stackoverflow.com/a/33533514
from __future__ import annotations

import copy
import dataclasses
import enum
import functools
import json
import operator
import typing


HIERARCHY_FILE = 'hardware_model_hierarchy.json'


# FIXME: complete list
class NvidiaGPUComponentEnum(enum.Enum):
    NONE = enum.auto()
    GraphicProcessingCluster = enum.auto()
    RasterOperatorPartition = enum.auto()
    RasterOperator = enum.auto()
    TextureProcessingCluster = enum.auto()
    StreamingMultiprocessor = enum.auto()
    CUDACore = enum.auto()


# https://stackoverflow.com/a/24482806
# we need a specialized parser for saving and loading the enumerations
class GPUComponentEnumJSONConverter(json.JSONEncoder):
    GPUComponentEnumList = {NvidiaGPUComponentEnum.__name__: NvidiaGPUComponentEnum}

    # this is the encoder, to map correctly the Enum classes
    def default(self, obj):
        if type(obj) in self.GPUComponentEnumList.values():
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)

    # this is the decoder, to convert back into Python objects
    @staticmethod
    def decode(d):
        if "__enum__" in d:
            name, member = d["__enum__"].split(".")
            return getattr(GPUComponentEnumJSONConverter.GPUComponentEnumList[name],
                           member)
        else:
            return d


class GPUComponent(typing.NamedTuple):
    number_of_components: int
    component_type: NvidiaGPUComponentEnum
    parent: NvidiaGPUComponentEnum
    subcomponents: typing.List[NvidiaGPUComponentEnum, ...]


@dataclasses.dataclass(init=True, repr=True)
class Kernel(object):
    thread_block_size: typing.Tuple[int, ...]
    # input and output sizes have a tuple containing dimension tuples
    # ((1, 2), (3, 4)) means that we have two inputs or outputs, the first with
    # dimensions 1 and 2 and the second with 3 and 4
    input_size: typing.Tuple[typing.Tuple[int, ...], ...]
    # output can be only one, but size can change
    output_size: typing.Tuple[int, ...]
    # similar but for each thread
    thread_input_size: typing.Tuple[typing.Tuple[int, ...], ...]
    # output can be only one, but size can have multiple dimensions
    thread_output_size: typing.Tuple[int, ...]

    # number of threads required to compute the total result
    # computed dinamically as it depends on kernel and thread output sizes
    # we compute the total number of output elements and we divide it buy the
    # number of output elements we obtain from each thread
    @property
    def n_threads(self):
        n_output_elements = functools.reduce(operator.mul, self.output_size)
        n_thread_output_elements = functools.reduce(operator.mul,
                                                    self.thread_output_size)
        return n_output_elements / n_thread_output_elements


@dataclasses.dataclass(init=True, repr=True)
class HardwareModel(object):
    # string of hierarchy taken from the JSON file
    json_hierarchy_string = dataclasses.field(init=True, repr=False)
    # string of map with all the enum values, from the JSON file
    json_map_string = dataclasses.field(init=True, repr=False)

    # parsed hierarchy
    _hierarchy = None
    # parsed map
    _map = None
    # maximum number of parallel threads
    _max_parallel_threads = None

    def __post_init__(self):
        self._hierarchy = self._parse_hierarchy(hierarchy=self.json_hierarchy_string,
                                                data_class=GPUComponent)
        self._map = self._parse_map(map_=self.json_map_string)
        self._max_parallel_threads = self._count_parallel_threads(self._hierarchy,
                                        NvidiaGPUComponentEnum.CUDACore)

    def _parse_hierarchy(self, hierarchy, data_class):
        hierarchy = json.loads(hierarchy,
                               object_hook=GPUComponentJEnumSONConverter.decode)
        parsed_hierarchy = {}
        for enum_value, c in hierarchy.items():
            component = data_class(*c)
            parsed_hierarchy[enum_value] = component
        return parsed_hierarchy

    def _parse_map(self, map_):
        map_ = json.loads(map_,
                          object_hook=GPUComponentEnumJSONConverter.decode)
        return map_

    # to compute the number for _parallel_threads
    def _count_parallel_threads(self, hierarchy, target):
        # we go through all the components to get the total number
        # we start by setting the parent, where we loop, to the current target
        # module
        parent = hierarchy[target]
        component_count = 1
        # we stop when we find a parent to be none
        while parent.parent is not NvidiaGPUComponentEnum.NONE:
            parent = hierarchy[parent.parent]
            component_count *= parent.component_count
        return component_count

    @property
    def hierarchy(self):
        return copy.deepcopy(self._hierarchy)

    @property
    def map(self):
        return copy.deepcopy(self._map)
