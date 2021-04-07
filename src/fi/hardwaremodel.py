# for forward annotations, when using the defined class inside the class itself
# for annotations
# only required for Python 3.7 up to 3.9, standard from 3.10
# for Python 3.6 and earlier, use a string with the name of the class
# https://stackoverflow.com/a/33533514
from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import functools
import json
import math
import operator
import typing

from . import summary


# FIXME: complete list
class NvidiaGPUComponentEnum(enum.Enum):
    NONE = enum.auto()
    GraphicProcessingCluster = enum.auto()
    RasterEngine = enum.auto()
    RasterOperatorPartition = enum.auto()
    RasterOperator = enum.auto()
    TextureProcessingCluster = enum.auto()
    StreamingMultiprocessor = enum.auto()
    StreamingMultiprocessorPartition = enum.auto()
    CUDACore = enum.auto()
    FP32Datapath = enum.auto()
    FP32INT32Datapath = enum.auto()
    RayTracingCore = enum.auto()
    RegisterFile = enum.auto()
    TensorCore = enum.auto()
    TextureUnit = enum.auto()
    L0InstructionCache = enum.auto()
    WarpScheduler = enum.auto()
    DispatchUnit = enum.auto()
    LoaDSToreUnit = enum.auto()
    SpecialFunctionUnit = enum.auto()
    FP64Datapath = enum.auto()
    L1InstructionCacheSharedMemory = enum.auto()
    PolyMorphEngine = enum.auto()


# https://stackoverflow.com/a/24482806
# we need a specialized parser for saving and loading the enumerations
class GPUComponentEnumJSONConverter(json.JSONEncoder):
    GPUComponentEnumList = {NvidiaGPUComponentEnum.__name__:
                                                        NvidiaGPUComponentEnum}

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


class GPUComponentHierarchy(typing.NamedTuple):
    # the number of components existing in each parent instance, so if we have
    # the structure A contains B, and we have 2 A instances and 2 B per parent
    # instance, we get a total of 4 B instances in the whole device
    # in some particular cases, the number of components per parent can be a
    # float, to represent components which are available only in a subset of
    # parents
    number_of_components_per_parent: typing.Union[int, float]
    component_type: NvidiaGPUComponentEnum
    parent: NvidiaGPUComponentEnum
    subcomponents: typing.List[NvidiaGPUComponentEnum, ...]


class GPUComponent(typing.NamedTuple):
    # we generally use sequential component ids
    component_id: int
    component_type: NvidiaGPUComponentEnum


# FIXME: implement all kernels
@dataclasses.dataclass(init=True, repr=True)
class Kernel(object):
    # for Nvidia A100, we have 32 threads per warp
    THREADS_PER_WARP = 32

    # input and output sizes have a tuple containing dimension tuples
    input_size: typing.Tuple[int, ...]
    # output can be only one, but size can change
    output_size: typing.Tuple[int, ...]
    # weight can be only one, with different sizes
    weight_size: typing.Tuple[int, ...]
    # bias size, only one
    bias_size: typing.Tuple[int, ...]
    # similar but for each thread
    thread_input_size: typing.Tuple[int, ...]
    # output can be only one, but size can have multiple dimensions
    thread_output_size: typing.Tuple[int, ...]
    # similar but for each thread
    thread_weight_size: typing.Tuple[int, ...]
    # output can be only one, but size can have multiple dimensions
    thread_bias_size: typing.Tuple[int, ...]

    # number of threads required to compute the total result
    # computed dinamically as it depends on kernel and thread output sizes
    # we compute the total number of output elements and we divide it buy the
    # number of output elements we obtain from each thread
    @property
    def n_threads(self) -> int:
        n_output_elements = functools.reduce(operator.mul, self.output_size)
        n_thread_output_elements = functools.reduce(operator.mul,
                                                    self.thread_output_size)
        return math.ceil(n_output_elements / n_thread_output_elements)

    @property
    def n_warps(self) -> int:
        return math.ceil(self.n_threads / self.THREADS_PER_WARP)

    # FIXME: implement a way of handling different blocks, this could be useful
    # for handling memory conflicts in the hardware model
    @property
    def n_blocks(self):
        return NotImplemented

    # this function returns a dict containing the association between a target
    # id and the corresponding thread to be run
    # it takes care of creating the thread descriptors with the correct
    # parameters and saving all of them
    def make_threads(self, target_ids: typing.Tuple[int, ...],
                     time_start: float, time_stop: float,
                     total_time: float) -> typing.Dict[int, ThreadDescriptor]:
        threads = {}  # target id: thread
        for target_id, thread_id in zip(target_ids, range(self.n_threads)):
            thread = ThreadDescriptor(
                                      thread_id=thread_id,
                                      target_id=target_id,
                                      parent_kernel=self,
                                      time_start=time_start,
                                      time_stop=time_stop,
                                      total_time=total_time,
                                      )
            threads[target_id] = thread
        return threads


# thread descriptor class
class ThreadDescriptor(typing.NamedTuple):
    thread_id: int
    target_id: int
    parent_kernel: Kernel
    time_start: float
    time_stop: float
    total_time: float


# FIXME: complete the sample hierarchy
SAMPLE_HIERARCHY = [
    {
        'number_of_components_per_parent': 7,
        'component_type': NvidiaGPUComponentEnum.GraphicProcessingCluster,
        'parent': NvidiaGPUComponentEnum.NONE,
        'subcomponents': [
            NvidiaGPUComponentEnum.RasterOperatorPartition,
        ],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.RasterEngine,
        'parent': NvidiaGPUComponentEnum.GraphicProcessingCluster,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 2,
        'component_type': NvidiaGPUComponentEnum.RasterOperatorPartition,
        'parent': NvidiaGPUComponentEnum.GraphicProcessingCluster,
        'subcomponents': [
            NvidiaGPUComponentEnum.RasterOperator,
        ],
    },
    {
        'number_of_components_per_parent': 8,
        'component_type': NvidiaGPUComponentEnum.RasterOperator,
        'parent': NvidiaGPUComponentEnum.RasterOperatorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 6,
        'component_type': NvidiaGPUComponentEnum.TextureProcessingCluster,
        'parent': NvidiaGPUComponentEnum.GraphicProcessingCluster,
        'subcomponents': [
            NvidiaGPUComponentEnum.PolyMorphEngine,
            NvidiaGPUComponentEnum.StreamingMultiprocessor,
            ],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.PolyMorphEngine,
        'parent': NvidiaGPUComponentEnum.TextureProcessingCluster,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 2,
        'component_type': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'parent': NvidiaGPUComponentEnum.TextureProcessingCluster,
        'subcomponents': [
            NvidiaGPUComponentEnum.RayTracingCore,
            NvidiaGPUComponentEnum.L1InstructionCacheSharedMemory,
            NvidiaGPUComponentEnum.FP64Datapath,
            NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
            ],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.RayTracingCore,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type':
        NvidiaGPUComponentEnum.L1InstructionCacheSharedMemory,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 2,
        'component_type': NvidiaGPUComponentEnum.FP64Datapath,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 4,
        'component_type':
        NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'subcomponents': [
            NvidiaGPUComponentEnum.TensorCore,
            NvidiaGPUComponentEnum.RegisterFile,
            NvidiaGPUComponentEnum.TextureUnit,
            NvidiaGPUComponentEnum.L0InstructionCache,
            NvidiaGPUComponentEnum.WarpScheduler,
            NvidiaGPUComponentEnum.DispatchUnit,
            NvidiaGPUComponentEnum.LoaDSToreUnit,
            NvidiaGPUComponentEnum.SpecialFunctionUnit,
            NvidiaGPUComponentEnum.CUDACore,
            ],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.TensorCore,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.RegisterFile,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.TextureUnit,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.L0InstructionCache,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.WarpScheduler,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.DispatchUnit,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 4,
        'component_type': NvidiaGPUComponentEnum.LoaDSToreUnit,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.SpecialFunctionUnit,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 32,
        'component_type': NvidiaGPUComponentEnum.WarpScheduler,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [
            NvidiaGPUComponentEnum.FP32Datapath,
            NvidiaGPUComponentEnum.FP32INT32Datapath,
        ],
    },
    {
        'number_of_components_per_parent': 0.5,
        'component_type': NvidiaGPUComponentEnum.FP32Datapath,
        'parent': NvidiaGPUComponentEnum.CUDACore,
        'subcomponents': [],
    },
    {
        'number_of_components_per_parent': 0.5,
        'component_type': NvidiaGPUComponentEnum.FP32INT32Datapath,
        'parent': NvidiaGPUComponentEnum.CUDACore,
        'subcomponents': [],
    },
]

# FIXME: complete the sample map
SAMPLE_MAP = [
    [
        [],
        [],
    ],
    [
        [],
        [],
    ],
]



@dataclasses.dataclass(init=True, repr=True)
class HardwareModel(object):
    # string of hierarchy taken from the JSON file
    json_hierarchy_string: str = dataclasses.field(init=True, repr=False)
    # string of map with all the enum values, from the JSON file
    json_map_string: str = dataclasses.field(init=True, repr=False)

    # parsed hierarchy, basically a list of dicts with the arguments for
    # GPUComponent
    _hierarchy = None
    # parsed map
    _map = None
    # maximum number of parallel threads
    _max_parallel_threads = None

    def __post_init__(self):
        self._hierarchy = self._parse_hierarchy(
                            hierarchy=self.json_hierarchy_string,
                            data_class=GPUComponentHierarchy)
        self._map = self._parse_map(map_=self.json_map_string,
                                    data_class=GPUComponent)
        self._max_parallel_threads = self._count_max_parallel_threads(
                                        self._hierarchy)

    def _parse_hierarchy(self, hierarchy, data_class):
        hierarchy = json.loads(
                        hierarchy,
                        object_hook=GPUComponentJEnumSONConverter.decode)
        parsed_hierarchy = {}
        for c in hierarchy:
            component = data_class(**c)
            parsed_hierarchy[c.component_type] = component
        return parsed_hierarchy

    def _parse_map(self, map_, data_class):
        map_ = json.loads(map_,
                          object_hook=GPUComponentEnumJSONConverter.decode)
        # we assume map is 2D
        map_array = []
        for row in map_:
            parsed_row = []
            for cell in row:
                parsed_cell = []
                for component in cell:
                    parsed_cell.append(data_class(**component))
                parsed_row.append(parsed_cell)
            map_array.append(parsed_row)

        return map_array

    # to compute the number of components by traversing through the hierarchy
    # from bottom to top
    def _count_number_of_components_per_device(self, hierarchy, target):
        # we go through all the components to get the total number
        # we start by setting the parent, where we loop, to the current target
        # module
        parent = hierarchy[target]
        component_count = parent.number_of_components_per_parent
        # we stop when we find a parent to be none
        while parent.parent is not NvidiaGPUComponentEnum.NONE:
            parent = hierarchy[parent.parent]
            component_count *= parent.number_of_components_per_parent
        return math.ceil(component_count)

    def _count_max_parallel_threads(self, hierarchy):
        return self._count_number_of_components_per_device(hierarchy,
                                            NvidiaGPUComponentEnum.CUDACore)

    # FIXME: add defaults for target target ids
    # NOTE: in the future we can add support for custom free operators, if we
    # want to simulate a double parallel run or something that occupies some
    # operators in a certain order, using an extra arguments used as baseline
    # for the free_target_operators variable
    # by default we assume all of the target operators are available at time 0
    # and we also assume their number is contiguous
    def schedule_model_inference_run(self,
                                     summary_: summary.Summary,
                                     target: NvidiaGPUComponentEnum,
                                     ) -> typing.Dict[int,
                                        typing.List[ThreadDescriptor, ...]]:
        # NOTE: we assume that defaultdict is ordered, so Python 3.7+
        # first we compute the list of all the possible CUDA cores for
        # scheduling
        # so while we left choice for NvidiaGPUComponentEnum, it is supposed
        # to be indicating a CUDA core or equivalent
        schedule_dict = {}  # target id: list of ThreadDescriptor
        # we use math.ceil as the number may be a float
        target_components = math.ceil(
                                self._count_number_of_components_per_device(
                                    self._hierarchy, target))

        # first of all, we need to go through the list of the kernels to be
        # processed, to create the corresponding list of kernels with threads
        # information
        # this is required as kernel times are relative, this value is in
        # seconds
        total_execution_time = summary_.total_execution_time

        kernels = {}  # layer summary: kernel
        for layer in summary_.layer_stats:
            kernel = Kernel(
                            input_size=layer.input_size,
                            output_size=layer.output_size,
                            weight_size=layer.weight_size,
                            bias_size=layer.bias_size,
                            thread_output_size=(1, ),
                            # the following ones are not used for now
                            thread_weight_size=tuple(),
                            thread_bias_size=tuple(),
                            )
            kernels[layer] = kernel

        # now that we have all the kernels associated to their layers, we need
        # to schedule them, based on the required CUDA cores
        # we assume the operators run in order as the layers
        # at the beginning we set all the target ids to be free
        free_target_operators = {}  # timestamp: list of target id
        free_target_operators[0.0] = list(range(target_components))
        for layer, kernel in kernels.items():
            # we get the number of required operators, which is the same as
            # the number of threads to be run in the kernel
            n_required_operators = kernel.n_threads

            # we initialize the current time index to 0, the first key in the
            # dict
            current_time_index = 0
            # we update the current time to current index
            current_time = list(free_target_operators.keys())[
                                                        current_time_index]
            # we have a list of the operators which are free at the current
            # time
            current_free_operators = free_target_operators[current_time]
            # we select the subset of operators
            # we need to check whether we have enough operators available at
            # current time, otherwise we reach the next time when more
            # operators are freed
            # while the above algorithm is a good solution, it would have very
            # low overall utilization, leading to much worse performance with
            # lower power usage
            # therefore, we have to select the maximum possible subset of
            # operators at each time slot, and update the dict accordingly

            # we also have a temporary dict for updating the number of
            # operators number of totally selected operators, must match the
            # required number
            n_selected_operators = 0

            # we compute the execution time for the threads, we assume a thread
            # takes the whole kernel time as they are all executed in parallel,
            # with no memory conflicts
            execution_time = total_execution_time * layer.relative_execution_time

            while n_required_operators > n_selected_operators:
                # we get the number of operators from the current free
                # operators
                # in Python, if we overflow the end of the list splice, we get
                # all the elements until the end, so there are no IndexErrors
                # however we must check this number before continuing
                selected_operators = current_free_operators[:(n_required_operators - n_selected_operators)]
                # we update the set of current free operators, by removing the
                # selected ones
                free_target_operators[current_time] = current_free_operators[(n_required_operators - n_selected_operators):]
                # we update the future free target operators, so we take the
                # current existing list of operators and we add the operators
                # which will free when the execution of the current threads is
                # completed
                # if the time does not exist, then we set all of them to be
                # free, as it means no other scheduling has occurred yet
                if current_time + execution_time not in free_target_operators:
                    free_target_operators[current_time + execution_time] = list(range(target_components))
                # otherwise we append the free operators and we sort them
                else:
                    free_target_operators[current_time + execution_time].extend(selected_operators)
                    free_target_operators[current_time + execution_time].sort()

                # we create the threads and we insert them in the schedule dict
                # to be returned
                threads = kernel.make_threads(target_ids=selected_operators,
                                              start_time=current_time,
                                              stop_time=current_time + execution_time,
                                              total_time=execution_time)
                for target_id, thread_desc in threads.values():
                    if target_id not in schedule_dict:
                        schedule_dict[target_id] = []
                    schedule_dict[target_id].append(thread_desc)

                # we update the count of the selected operators
                n_selected_operators += len(selected_operators)
                # we add the new threads on the list
                # we update the current time index
                current_time_index += 1
                # we update all the current time pointers
                current_time = list(free_target_operators.keys())[current_time_index]
                current_free_operators = free_target_operators[current_time]
        # once done we return the final scheduling
        return schedule_dict

    # FIXME: add return type annotation
    @property
    def hierarchy(self):
        return copy.deepcopy(self._hierarchy)

    # FIXME: add return type annotation
    @property
    def map(self):
        return copy.deepcopy(self._map)
