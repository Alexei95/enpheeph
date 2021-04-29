import ast
import collections
import copy
import dataclasses
import enum
import io
import json
import typing

import numpy
import onnx
import onnxruntime
import torch
import torchprof
import torchinfo


# this enum represents the main function layer
class MainLayerFunctionEnum(enum.Enum):
    Conv2d = enum.auto()
    MaxPool2d = enum.auto()
    AdaptiveAvgPool2d = enum.auto()
    ReLU = enum.auto()
    Dropout = enum.auto()
    Linear = enum.auto()

    # we can construct the enum from the string, since the name are the same as
    # the main layer functions in PyTorch
    @classmethod
    def from_string(cls, string: str):
        # we cover for this exception to return a more meaningful one
        # calling the name of the enum
        try:
            return cls.__members__[string]
        except KeyError:
            raise ValueError(
                    f'{string} is not a valid {cls.__name__} enum value'
                    )


# we converted it to dataclass to allow for more flexibility with properties,
# arguments and methods
# we need a hashable class as it is used as key for dicts with kernels
# this automated hash implementation is unsafe, as it does not take into
# account modifiability, but it can be used as long as the object does not
# change
@dataclasses.dataclass(init=True, repr=True, unsafe_hash=True)
class LayerInfo(object):
    # used to identify the main kernel type
    kernel_type: MainLayerFunctionEnum
    # used to have intrinsic ordering
    index: int
    # a string with the representation of the layer
    # it contains other useful info which must be parsed on a layer-by-layer
    # basis
    representation: str
    # here we parse the representation string, as a dict
    _parsed_representation = None
    # input size, including also batch size as first value
    input_size: typing.Tuple[int, ...]
    # output size, including also batch size as first value
    output_size: typing.Tuple[int, ...]
    # weightsize, not used for some layers, empty in those cases
    # for fully connected layers this containes the matrix size for the weights
    # for convolutional layers, this contains the size of the kernel
    weight_size: typing.Tuple[int, ...]
    # bias size, empty if no bias
    bias_size: typing.Tuple[int, ...]
    # execution time relative to the total one, between 0 and 1
    relative_execution_time: float
    # original execution time, as reported by the profiler
    original_absolute_execution_time: float

    def __post_init__(self):
        self._parsed_representation = self.parse_representation(
            self.representation
            )

    @property
    def parsed_representation(self) -> typing.Dict[str, typing.Any]:
        return copy.deepcopy(self._parsed_representation)

    # this method parses the representation using AST, the Python grammar
    # in this way it is safe to execute compared to eval, and provides the
    # same level of detail
    # we parse sequential arguments with the key __args__ as a list, while all
    # the other keywords are mapped 1-to-1
    # we assume that the representation is a Python function call of the form
    # Layer(arg1, arg2, a=1, b=(1, 2), ...)
    # and it will return {'__args__': [arg1, arg2], 'a': 1, 'b': (1, 2)}
    # if there are multiple calls, only the first one will be considered
    # the idea is to isolate the function call and use ast.literal_eval to
    # return the Python values of each argument
    # taken from https://stackoverflow.com/a/49723227
    @staticmethod
    def parse_representation(representation: str) -> typing.Dict[
                                                            str, typing.Any]:
        # the function call object is inside the tree
        func_call = ast.parse(representation).body[0].value
        # we initialize the representation to the empty sequential argument
        # list
        parsed_repr = {'__args__': []}

        # we iterate over the sequential arguments
        for arg in func_call.args:
            # we append the value of each argument execution to the list
            parsed_repr['__args__'].append(ast.literal_eval(arg))
        # we iterate over the keyword arguments
        for kwarg in func_call.keywords:
            # in kwarg.arg there is the name of the argument, while in
            # kwarg.value there is the value which must be eval'd
            parsed_repr[kwarg.arg] = ast.literal_eval(kwarg.value)

        return parsed_repr


# this class simply works as a wrapper for the summaries required for computing
# the different parts of the model
# we mostly need operation-wise execution time (which can be relaxed to
# layer-wise, assuming one main operation per layer)
@dataclasses.dataclass(init=True)
class NNModelSummary(object):
    model: torch.nn.Module
    # input size must have the first element as batch size, which can be also 1
    input_size: typing.Sequence[int, ...]

    # if we don't want fields, just put normal values
    _torchinfo_summary = None
    _torchprof_layer_profiling = None
    _layer_stats = None
    _total_execution_time = None  # seconds

    def __post_init__(self):
        # we initialize the internal layer stats
        self._layer_stats = []

        # we populate the layer stats
        # summary is computed in here
        # profiling is done inside this function, one layer at a time
        # therefore we don't have raw profiling data in a single object
        self._populate_layer_stats()

    def _populate_layer_stats(self, force_update=False):
        # we check if it has been already updated
        if self._layer_stats and not force_update:
            return

        # this dummy input is used for the summary over the whole model
        dummy_input = torch.randn(*self.input_size)
        summary = torchinfo.summary(self.model, input_data=dummy_input)

        self._torchinfo_summary = summary

        layers = [layer
                  for layer in self._torchinfo_summary.summary_list
                  if not layer.inner_layers]

        # we set up the layer profiling results, each key is the summary of the
        # layer, while the item is the profiling result itself
        layer_profiling = collections.OrderedDict()
        # we cycle through all the leaf layers
        for layer in layers:
            # we create copies on the GPU of the module and a dummy input
            module = copy.deepcopy(layer.module).eval().to('cuda')
            dummy_input = torch.randn(*layer.input_size, device='cuda')

            # we profile with no backward accumulation
            with torchprof.Profile(module, enabled=True, use_cuda=True,
                                   profile_memory=True) as prof:
                with torch.no_grad():
                    module(dummy_input)

            # we save a copy of the results
            layer_profiling[layer] = copy.deepcopy(prof)

        self._torchprof_layer_profiling = copy.deepcopy(layer_profiling)

        # we compute the total execution time on CUDA by going over all
        # possible events, and summing over the total CUDA time of each of them
        # the result is in microseconds
        # the structure is
        # layer_profile -> trace-event list dict -> sub-event list -> event
        # pay attention as trace_profile_events is a defaultdict, so to get
        # the time we need to go over the values
        total_execution_time = sum(
            e.cuda_time_total
            for p in self._torchprof_layer_profiling.values()
            for event_list in p.trace_profile_events.values()
            for subevent_list in event_list
            for e in subevent_list)

        # remember it's in microseconds, so we convert to seconds
        self._total_execution_time = total_execution_time * 10 ** (-6)

        # now we have to make the data easily accessible
        for i, (lsummary, lprof) in enumerate(
                self._torchprof_layer_profiling.items()):
            # as for total time, we iterate over all event lists
            # in this case we are already iterating over the layers, so we skip
            # that
            # pay attention as trace_profile_events is a defaultdict, so to get
            # the time we need to go over the values
            # this time is in microseconds as well
            execution_time = sum(
                e.cuda_time_total
                for event_list in lprof.trace_profile_events.values()
                for subevent_list in event_list
                for e in subevent_list)
            # we make it relative, since total_execution_time and
            # execution_time are us, we can use them directly
            relative_execution_time = execution_time / total_execution_time

            # bias size, we get the bias, defaulting to None
            # we don't use an empty torch.Tensor as its size is [0]
            # but we would like it to be []
            bias = getattr(lsummary.module, 'bias', None)
            # since size is a callable, we use tuple as default, so that we can
            # call it to generate an empty element
            # we convert the size in a standard tuple
            bias_size = tuple(getattr(bias, 'size', tuple)())

            # we generate the object
            linfo = LayerInfo(
                kernel_type=MainLayerFunctionEnum.from_string(
                    lsummary.class_name
                    ),
                index=i,
                representation=repr(lsummary.module),
                input_size=tuple(lsummary.input_size),
                output_size=tuple(lsummary.output_size),
                weight_size=tuple(lsummary.kernel_size),
                bias_size=bias_size,
                relative_execution_time=relative_execution_time,
                # convert the original time to seconds
                original_absolute_execution_time=execution_time * 10 ** (-6),
                )

            # we append it to the stats list
            self._layer_stats.append(linfo)

    @property
    def layer_stats(self) -> typing.List[LayerInfo]:
        return copy.deepcopy(self._layer_stats)

    # return value is in seconds
    @property
    def total_execution_time(self) -> float:
        return self._total_execution_time

    @property
    def raw_torchinfo_summary(self) -> torchinfo.ModelStatistics:
        return copy.deepcopy(self._torchinfo_summary)

    @property
    def raw_torchprof_layer_profiling(self) -> typing.Dict[
                                            torchinfo.layer_info.LayerInfo,
                                            torchprof.Profile]:
        return copy.deepcopy(self._torchprof_layer_profiling)
