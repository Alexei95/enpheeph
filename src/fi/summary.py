import collections
import copy
import dataclasses
import io
import json
import typing

import numpy
import onnx
import onnxruntime
import torch
import torchprof
import torchinfo


class LayerInfo(typing.NamedTuple):
    # used to identify the main kernel type
    name: str
    # used to have intrinsic ordering
    index: int
    # a string with the representation of the layer
    # it contains other useful info which must be parsed on a layer-by-layer
    # basis
    representation: str
    # input size, including also batch size as first value
    input_size: typing.Tuple[int, ...]
    # output size, including also batch size as first value
    output_size: typing.Tuple[int, ...]
    # kernel size, not used for some layers, left blank in those cases
    kernel_size: typing.Tuple[int, ...]
    # execution time relative to the total one, between 0 and 1
    relative_execution_time: float


# this class simply works as a wrapper for the summaries required for computing
# the different parts of the model
# we mostly need operation-wise execution time (which can be relaxed to
# layer-wise, assuming one main operation per layer)
@dataclasses.dataclass(init=True)
class Summary(object):
    model: torch.nn.Module
    # input size must have the first element as batch size, which can be also 1
    input_size: typing.Tuple[int, ...]

    # if we don't want fields, just put normal values
    _torchinfo_summary = None
    _torchprof_layer_profiling = None
    _layer_stats = None
    _total_execution_time = None  # microseconds

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

        layers = [l
                  for l in self._torchinfo_summary.summary_list
                  if not l.inner_layers]

        # we set up the layer profiling results, each key is the summary of the
        # layer, while the item is the profiling result itself
        layer_profiling = collections.OrderedDict()
        # we cycle through all the leaf layers
        for l in layers:
            # we create copies on the GPU of the module and a dummy input
            module = copy.deepcopy(l.module).eval().to('cuda')
            dummy_input = torch.randn(*l.input_size, device='cuda')

            # we profile with no backward accumulation
            with torchprof.Profile(module, enabled=True, use_cuda=True,
                                   profile_memory=True) as prof:
                with torch.no_grad():
                    module(dummy_input)

            # we save a copy of the results
            layer_profiling[l] = copy.deepcopy(prof)

        self._torchprof_layer_profiling = copy.deepcopy(layer_profiling)

        # we compute the total execution time on CUDA by going over all
        # possible events, and summing over the total CUDA time of each of them
        # the result is in microseconds
        # the structure is
        # layer_profile -> trace-event list dict -> sub-event list -> event
        # pay attention as trace_profile_events is a defaultdict, so to get
        # the time we need to go over the values
        total_execution_time = sum(e.cuda_time_total
                            for p in self._torchprof_layer_profiling.values()
                            for event_list in p.trace_profile_events.values()
                            for subevent_list in event_list
                            for e in subevent_list)

        # now we have to make the data easily accessible
        for i, (lsummary, lprof) in enumerate(self._torchprof_layer_profiling.items()):
            # as for total time, we iterate over all event lists
            # in this case we are already iterating over the layers, so we skip
            # that
            # pay attention as trace_profile_events is a defaultdict, so to get
            # the time we need to go over the values
            execution_time = sum(e.cuda_time_total
                                for event_list in lprof.trace_profile_events.values()
                                for subevent_list in event_list
                                for e in subevent_list)
            # we make it relative
            relative_execution_time = execution_time / total_execution_time

            # we generate the object
            linfo = LayerInfo(name=lsummary.class_name,
                              index=i,
                              representation=repr(lsummary.module),
                              input_size=lsummary.input_size,
                              output_size=lsummary.output_size,
                              kernel_size=lsummary.kernel_size,
                              relative_execution_time=relative_execution_time,
                              )

            # we append it to the stats list
            self._layer_stats.append(linfo)

    @property
    def layer_stats(self) -> typing.List[LayerInfo]:
        return copy.deepcopy(self._layer_stats)

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
