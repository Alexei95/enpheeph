# for forward annotations, when using the defined class inside the class itself
# for annotations
# only required for Python 3.7 up to 3.9, standard from 3.10
# for Python 3.6 and earlier, use a string with the name of the class
# https://stackoverflow.com/a/33533514
from __future__ import annotations

import abc
import copy
import dataclasses
import enum
import functools
import json
import math
import operator
import pprint
import typing

import src.fi.modeling.nnmodelsummary


# https://stackoverflow.com/a/24482806
# we need a specialized parser for saving and loading the enumerations
# this class has been customized to support registering new encode/decode
# functions for different objects
# the structure is to have a element __cls__ in the dict, matching with the
# class name corresponding to the function
# in the encoder we use the class as index, as we receive Python objects
# in the decoder we use the class name, since it must be converted to string
# for being JSON-serializable
# we also keep a class to class name association dict
class JSONConverter(json.JSONEncoder):
    EncoderList = {}
    DecoderList = {}
    ClassNameAssociation = {}

    # the encoder should return a dict
    # the decoder should allow extra keys in the dict, as there is also __cls__
    # and it should return the corresponding Python object
    @classmethod
    def register_class(cls, class_, encode_func, decode_func):
        cls.EncoderList[class_] = encode_func
        # use qualname instead of name for classes, to allow nested classes
        cls.DecoderList[class_.__qualname__] = decode_func
        # use qualname instead of name for classes, to allow nested classes
        cls.ClassNameAssociation[class_] = class_.__qualname__

    # this function can be used as a wrapper on the class, assuming there are
    # two methods to_json and from_json to convert it
    @classmethod
    def register_class_decorator(cls, class_):
        cls.register_class(class_, class_.to_json, class_.from_json)
        return class_

    @classmethod
    def deregister_class(cls, class_):
        del cls.EncoderList[class_]
        # use qualname instead of name for classes, to allow nested classes
        del cls.DecoderList[class_.__qualname__]
        del cls.ClassNameAssociation[class_]

    # this is the encoder, to map correctly the registered classes
    def default(self, obj):
        # type returns the class if the obj is an instance of such class
        # be careful in not using classes, as they are of type 'type'
        if type(obj) in self.EncoderList:
            encoded_obj = self.EncoderList[type(obj)](obj)
            # since we are working with instances, we get the class defining
            # the object and we get its name, which should provide correct
            # results for both Enums and standard classes
            # use qualname instead of name for classes, to allow nested classes
            return {"__cls__": obj.__class__.__qualname__, **encoded_obj}
        return json.JSONEncoder.default(self, obj)

    # this is the decoder, to convert back into Python objects
    @classmethod
    def decode(cls, d):
        if "__cls__" in d:
            cls_name = d['__cls__']
            return cls.DecoderList[cls_name](d)
        else:
            return d


class JSONSerializableABC(abc.ABC):
    # NOTE: when not implemented we need to raise NotImplementedError in
    # abstract methods
    @classmethod
    @abc.abstractmethod
    def to_json(cls, obj):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def from_json(cls, d):
        raise NotImplementedError()


class JSONSerializableDictClass(JSONSerializableABC):
    @classmethod
    def to_json(cls, obj):
        return obj.__dict__

    @classmethod
    def from_json(cls, d):
        d_ = d.copy()
        if '__cls__' in d_:
            del d_['__cls__']
        return cls(**d_)


# we require this metaclass to avoid conflicts between EnumMeta and ABCMeta
# so we need to use this as metaclass for covering both ABC subclasses and
# Enum subclasses
class JSONSerializableEnumMeta(abc.ABCMeta, enum.EnumMeta):
    pass


class JSONSerializableEnum(JSONSerializableABC):
    @classmethod
    def to_json(cls, obj):
        # the name of the enum value can be accessed using name
        return {'__enum__': obj.name}

    @classmethod
    def from_json(cls, d):
        return getattr(cls, d['__enum__'])


@JSONConverter.register_class_decorator
class NvidiaGPUComponentEnum(
        JSONSerializableEnum,
        enum.Enum,
        metaclass=JSONSerializableEnumMeta):
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


# NOTE: unfortunately, for typing.NamedTuple there are issues with inheritance
# as it changes the metaclass and hence the inheritance sequence for the fields
# the proposed solution would be to use dataclasses
# we lose unpacking properties and some nice behaviours, but it shouldn't
# matter much
@JSONConverter.register_class_decorator
@dataclasses.dataclass(init=True, repr=True)
class GPUComponentHierarchy(JSONSerializableDictClass):
    # the number of components existing in each parent instance, so if we have
    # the structure A contains B, and we have 2 A instances and 2 B per parent
    # instance, we get a total of 4 B instances in the whole device
    # in some particular cases, the number of components per parent can be a
    # float, to represent components which are available only in a subset of
    # parents
    number_of_components_per_parent: typing.Union[int, float]
    component_type: NvidiaGPUComponentEnum
    parent: NvidiaGPUComponentEnum
    subcomponents: typing.Sequence[NvidiaGPUComponentEnum, ...]


@JSONConverter.register_class_decorator
@dataclasses.dataclass(init=True, repr=True)
class GPUComponent(JSONSerializableDictClass):
    # we generally use sequential component ids
    component_id: int
    component_type: NvidiaGPUComponentEnum


# FIXME: implement all kernels, after the interface is well defined
@dataclasses.dataclass(init=True, repr=True)
class Kernel(object):
    # for Nvidia A100, we have 32 threads per warp
    THREADS_PER_WARP = 32

    # kernel type, used to define some sizes if they are not defined in the
    # init
    kernel_type: src.fi.modeling.nnmodelsummary.MainLayerFunctionEnum
    # extra arguments, they are kernel dependent
    extra_args: typing.Dict[str, typing.Any]
    # input and output sizes have a tuple containing dimension tuples
    input_size: typing.Tuple[int, ...]
    # output can be only one, but size can change
    output_size: typing.Tuple[int, ...]
    # weight can be only one, with different sizes
    weight_size: typing.Tuple[int, ...]
    # bias size, only one
    bias_size: typing.Tuple[int, ...]
    # output can be only one, but size can have multiple dimensions
    thread_output_size: typing.Tuple[int, ...]
    # input thread sizes can be optional, as they can change depending on the
    # kernel type and total size of the operation
    # similar but for each thread
    thread_input_size: typing.Optional[typing.Tuple[int, ...]] = None
    # similar but for each thread
    thread_weight_size: typing.Optional[typing.Tuple[int, ...]] = None
    # output can be only one, but size can have multiple dimensions
    thread_bias_size: typing.Optional[typing.Tuple[int, ...]] = None
    # here we can save the corresponding layer info of the kernel
    raw_layer_info: typing.Optional[src.fi.modeling.nnmodelsummary.LayerInfo] = None

    # BUG: complete the function for different layers
    # given three parameters it returns the 4th one
    @staticmethod
    def conv_missing_size(input=None, kernel=None, stride=None, output=None):
        pass

    # BUG: the layer-wise input and bias and weight sizes can be defined later
    # when taking into account memory and dataflow locks
    def __post_init__(self):
        # BUG: example of selection for dimensions
        if self.kernel_type is src.fi.modeling.nnmodelsummary.MainLayerFunctionEnum.Conv2d:
            pass
        pass

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
    def make_threads(
            self,
            target_ids: typing.Tuple[int, ...],
            start_time: float, stop_time: float,
            total_time: float,
            starting_unique_thread_id: typing.Optional[int] = 0
            ) -> typing.Dict[int, ThreadDescriptor]:
        threads = {}  # target id: thread
        for target_id, thread_id in zip(target_ids, range(self.n_threads)):
            thread = ThreadDescriptor(
                # this covers the local thread id, inside the current kernel
                kernel_thread_id=thread_id,
                # this is a sequential thread identifier used for GPU-wide
                # purposes
                unique_thread_id=thread_id + starting_unique_thread_id,
                target_id=target_id,
                parent_kernel=self,
                start_time=start_time,
                stop_time=stop_time,
                total_time=total_time,
                )
            threads[target_id] = thread
        return threads


# thread descriptor class
class ThreadDescriptor(typing.NamedTuple):
    # this id is local to the kernel
    kernel_thread_id: int
    # this id is global to the GPU
    unique_thread_id: int
    # the id of the target component where this thread will run
    target_id: int
    # the object with the parent kernel
    parent_kernel: Kernel
    # start, stop and total time taken for running the thread
    start_time: float
    stop_time: float
    total_time: float


# NOTE: based on Nvidia GPU A100, Ampere workstation GPU, 2020
SAMPLE_HIERARCHY = [
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 7,
        'component_type': NvidiaGPUComponentEnum.GraphicProcessingCluster,
        'parent': NvidiaGPUComponentEnum.NONE,
        'subcomponents': [
            NvidiaGPUComponentEnum.RasterOperatorPartition,
        ],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.RasterEngine,
        'parent': NvidiaGPUComponentEnum.GraphicProcessingCluster,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 2,
        'component_type': NvidiaGPUComponentEnum.RasterOperatorPartition,
        'parent': NvidiaGPUComponentEnum.GraphicProcessingCluster,
        'subcomponents': [
            NvidiaGPUComponentEnum.RasterOperator,
        ],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 8,
        'component_type': NvidiaGPUComponentEnum.RasterOperator,
        'parent': NvidiaGPUComponentEnum.RasterOperatorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 6,
        'component_type': NvidiaGPUComponentEnum.TextureProcessingCluster,
        'parent': NvidiaGPUComponentEnum.GraphicProcessingCluster,
        'subcomponents': [
            NvidiaGPUComponentEnum.PolyMorphEngine,
            NvidiaGPUComponentEnum.StreamingMultiprocessor,
            ],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.PolyMorphEngine,
        'parent': NvidiaGPUComponentEnum.TextureProcessingCluster,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 2,
        'component_type': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'parent': NvidiaGPUComponentEnum.TextureProcessingCluster,
        'subcomponents': [
            NvidiaGPUComponentEnum.RayTracingCore,
            NvidiaGPUComponentEnum.L1InstructionCacheSharedMemory,
            NvidiaGPUComponentEnum.FP64Datapath,
            NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
            ],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.RayTracingCore,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type':
        NvidiaGPUComponentEnum.L1InstructionCacheSharedMemory,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 2,
        'component_type': NvidiaGPUComponentEnum.FP64Datapath,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessor,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
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
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.TensorCore,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.RegisterFile,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.TextureUnit,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.L0InstructionCache,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.WarpScheduler,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.DispatchUnit,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 4,
        'component_type': NvidiaGPUComponentEnum.LoaDSToreUnit,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 1,
        'component_type': NvidiaGPUComponentEnum.SpecialFunctionUnit,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 32,
        'component_type': NvidiaGPUComponentEnum.CUDACore,
        'parent': NvidiaGPUComponentEnum.StreamingMultiprocessorPartition,
        'subcomponents': [
            NvidiaGPUComponentEnum.FP32Datapath,
            NvidiaGPUComponentEnum.FP32INT32Datapath,
        ],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 0.5,
        'component_type': NvidiaGPUComponentEnum.FP32Datapath,
        'parent': NvidiaGPUComponentEnum.CUDACore,
        'subcomponents': [],
    }),
    GPUComponentHierarchy(**{
        'number_of_components_per_parent': 0.5,
        'component_type': NvidiaGPUComponentEnum.FP32INT32Datapath,
        'parent': NvidiaGPUComponentEnum.CUDACore,
        'subcomponents': [],
    }),
]
SMALL_SAMPLE_HIERARCHY = list(map(
    lambda x: GPUComponentHierarchy(**{**x.__dict__,
                                       'number_of_components_per_parent': 1}),
    SAMPLE_HIERARCHY))
SAMPLE_HIERARCHY_JSON = json.dumps(
    SAMPLE_HIERARCHY,
    cls=JSONConverter,
    indent=4,
)
SMALL_SAMPLE_HIERARCHY_JSON = json.dumps(
    SMALL_SAMPLE_HIERARCHY,
    cls=JSONConverter,
    indent=4,
)

# FIXME: complete the sample map
SAMPLE_MAP = [
    [
        [
            {'component_id': 0,
             'component_type': NvidiaGPUComponentEnum.GraphicProcessingCluster,
             },
        ],
        [],
    ],
    [
        [],
        [],
    ],
]
SAMPLE_MAP_JSON = json.dumps(
    SAMPLE_MAP,
    cls=JSONConverter,
    indent=4,
)


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
                            hierarchy=self.json_hierarchy_string)
        self._map = self._parse_map(map_=self.json_map_string)
        self._max_parallel_threads = self._count_max_parallel_threads(
                                        self._hierarchy)

    def _parse_hierarchy(self, hierarchy: str):
        hierarchy = json.loads(
                        hierarchy,
                        object_hook=JSONConverter.decode)
        parsed_hierarchy = {}
        for c in hierarchy:
            parsed_hierarchy[c.component_type] = copy.deepcopy(c)
        return parsed_hierarchy

    def _parse_map(self, map_: str):
        map_ = json.loads(map_,
                          object_hook=JSONConverter.decode)
        # we assume map is 2D
        map_array = []
        for row in map_:
            parsed_row = []
            for cell in row:
                parsed_cell = []
                for component in cell:
                    parsed_cell.append(copy.deepcopy(component))
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
        while parent.parent is not parent.parent.NONE:
            parent = hierarchy[parent.parent]
            component_count *= parent.number_of_components_per_parent
        return math.ceil(component_count)

    def _count_max_parallel_threads(self, hierarchy):
        return self._count_number_of_components_per_device(
            hierarchy,
            NvidiaGPUComponentEnum.CUDACore)

    # FIXME: add defaults for target target ids
    # NOTE: in the future we can add support for custom free operators, if we
    # want to simulate a double parallel run or something that occupies some
    # operators in a certain order, using an extra arguments used as baseline
    # for the free_target_operators variable
    # by default we assume all of the target operators are available at time 0
    # and we also assume their number is contiguous
    def schedule_model_inference_run(
            self,
            model_summary: src.fi.modeling.nnmodelsummary.NNModelSummary,
            target: NvidiaGPUComponentEnum,
            ) -> typing.Dict[
                int,
                typing.Sequence[ThreadDescriptor]]:
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
        # we initialize the scheduling dict with all the components to an
        # empty list
        for target_component_id in range(target_components):
            schedule_dict[target_component_id] = []

        # first of all, we need to go through the list of the kernels to be
        # processed, to create the corresponding list of kernels with threads
        # information
        # this is required as kernel times are relative, this value is in
        # seconds
        total_execution_time = model_summary.total_execution_time

        kernels = {}  # layer summary: kernel
        for layer in model_summary.layer_stats:
            kernel = Kernel(
                            kernel_type=layer.kernel_type,
                            extra_args=layer.parsed_representation,
                            input_size=layer.input_size,
                            output_size=layer.output_size,
                            weight_size=layer.weight_size,
                            bias_size=layer.bias_size,
                            thread_output_size=(1, ),
                            raw_layer_info=layer,
                            )
            kernels[layer] = kernel

        # now that we have all the kernels associated to their layers, we need
        # to schedule them, based on the required CUDA cores
        # we assume the operators run in order as the layers
        # at the beginning we set all the target ids to be free
        free_target_operators = {}  # timestamp: list of target id
        free_target_operators[0.0] = list(range(target_components))

        # this variable is used to count the number of threads for giving
        # a progressive numbering
        # FIXME: improve progressive numbering of threads
        n_threads = 0

        # the current implementation resets the time index for each kernel,
        # leading to incorrect data dependency
        # the solution is to keep the time index constant across all
        # hence we have to initialize the time index before the loops, and keep
        # it coherent across the duration
        # we initialize the current time index to 0, the first key in the
        # dict
        current_time_index = 0
        for layer, kernel in kernels.items():
            # we get the number of required operators, which is the same as
            # the number of threads to be run in the kernel
            n_required_operators = kernel.n_threads

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
            execution_time = total_execution_time \
                * layer.relative_execution_time

            while n_required_operators > n_selected_operators:
                # we get the number of operators from the current free
                # operators
                # in Python, if we overflow the end of the list splice, we get
                # all the elements until the end, so there are no IndexErrors
                # however we must check this number before continuing
                selected_operators = current_free_operators[
                    :(n_required_operators - n_selected_operators)]
                # we update the set of current free operators, by removing the
                # selected ones
                free_target_operators[current_time] = current_free_operators[
                    (n_required_operators - n_selected_operators):]
                # we update the future free target operators, so we take the
                # current existing list of operators and we add the operators
                # which will free when the execution of the current threads is
                # completed
                # if the time does not exist, then we set all of them to be
                # free, as it means no other scheduling has occurred yet
                if current_time + execution_time not in free_target_operators:
                    free_target_operators[
                        current_time + execution_time] = list(
                            range(target_components))
                # otherwise we append the free operators and we sort them
                else:
                    free_target_operators[
                        current_time + execution_time].extend(
                            selected_operators)
                    free_target_operators[current_time + execution_time].sort()

                # we create the threads and we insert them in the schedule dict
                # to be returned
                threads = kernel.make_threads(
                    target_ids=selected_operators,
                    start_time=current_time,
                    stop_time=current_time + execution_time,
                    total_time=execution_time,
                    # we can use directly the number of threads as we start
                    # the numbering from 0
                    starting_unique_thread_id=n_threads)
                n_threads += len(threads)
                for target_id, thread_desc in threads.items():
                    schedule_dict[target_id].append(thread_desc)

                # we update the count of the selected operators
                n_selected_operators += len(selected_operators)
                # we add the new threads on the list
                # we update the current time index if there are no more free
                # operators left, so that we can access a new list in the next
                # kernel
                # otherwise we keep the current time index, but we update the
                # number of free operators
                if not free_target_operators[current_time]:
                    current_time_index += 1
                # we update all the current time pointers
                current_time = list(
                    free_target_operators.keys())[current_time_index]
                current_free_operators = free_target_operators[current_time]
            # before starting a new kernel, we need to update the time index to
            # match the first available point when **ALL** the threads from the
            # current kernel are done
            # to check, we simply check when the number of available targets is
            # the same as the total number, so when all of them are free
            # FIXME: in this way we lose some concurrency, but in the future
            # this could be fixed by splitting the thread dependencies, e.g.
            # we know how many of the previous threads must have finished to
            # start the new ones, based on number of inputs required for the
            # new threads and outputs produced by old ones
            for curr_time_idx, curr_free_operators in enumerate(
                    free_target_operators.values()
            ):
                if len(curr_free_operators) == target_components:
                    current_time_index = curr_time_idx


        # pprint.pprint(current_free_operators, indent=4)
        # pprint.pprint(free_target_operators, indent=4)
        # pprint.pprint(current_time, indent=4)
        # once done we return the final scheduling
        return schedule_dict

    def inject_fault(
            self,
            # we need to pass a schedule dict containing the ids of the targets
            # together with a list of the thread which are running
            schedule: typing.Dict[int, typing.List[ThreadDescriptor, ...]],
            # this argument represents the type of targets for the scheduling
            target: NvidiaGPUComponentEnum,
            # now we pass the fault description
            fault
    ):
        pass

    # FIXME: add return type annotation
    @functools.cached_property
    def hierarchy(self) -> typing.Dict[enum.Enum, GPUComponentHierarchy]:
        return copy.deepcopy(self._hierarchy)

    @functools.cached_property
    def hierarchy_json(self) -> str:
        return json.dumps(list(self._hierarchy.values()),
                          cls=JSONConverter,
                          indent=4)

    # FIXME: add return type annotation
    @functools.cached_property
    def map(self):
        return copy.deepcopy(self._map)
