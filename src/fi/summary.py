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
    name: str
    index: int
    input_size: typing.Tuple[int, ...]
    output_size: typing.Tuple[int, ...]
    extra_sizes: typing.Optional[typing.Tuple[typing.Tuple[int, ...]]]
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
    _torchprof_profiling = None
    _onnx_profiling = None
    _onnx_summary = None
    _layer_stats = None
    _total_execution_time = None  # microseconds

    def __post_init__(self):
        # we initialize the internal layer stats
        self._layer_stats = collections.OrderedDict()

        # we gather the layers info
        # self._get_torchinfo_summary()
        # self._get_onnx_summary()

        # we gather the execution times
        # self._get_torchprof_profiling()
        # self._get_onnx_profiling()

        # we populate the layer stats
        # profiling is done inside this function, one layer at a time
        self._populate_layer_stats()

    def _populate_layer_stats(self):
        # we check if it has been already updated
        if self._layer_stats:
            return

        print(self._torchprof_profiling.display(show_events=True))

        # we want only the leaf layers, which have no further children
        # to check further children we use the inner_layers, which returns a
        # dict
        # we do not need these, as we have high-level class info from the
        # profiler
        #layers = [l.class_name for l in self._torchinfo.summary_list
        #                       if not l.inner_layers]

        # we get the trace names from the profiling
        layer_traces = [t for t in self._torchprof_profiling.traces if t.leaf]
        # we get the path names from the traces
        layer_paths = [t.path for t in layer_traces]
        # we gather the corresponding events from the path names
        # each of this is actually a list, and it contains only one element
        # which is a EventList, having CPU/CUDA self-time
        # for the total time, it must be computed from sum of each event
        layer_events_container = [self._torchprof_profiling.trace_profile_events[p] for p in layer_paths]
        # we remove the extra list, so that we have only EventLists
        layer_event_lists = [e[0] for e in layer_events_container]
        # we sum all the events to get the total CUDA time
        # each event list can contain different sublists depending on the number
        # of children
        # FIXME: this works only with CUDA for now, must be adapted for CPU later
        layer_total_cuda_times = [sum(e.cuda_time for e in event_sublist)
                                  for event_list in layer_events_container
                                  for event_sublist in event_list]

        # we normalize the layer time, so that we are able to scale it to
        # different models,
        # the size scaling is done in the hardware model itself, while timing
        # may not be accurate if done on very different hardware
        self._total_execution_time = sum(layer_total_cuda_times)
        layer_rel_cuda_times = [cuda_time / self._total_execution_time
                                for cuda_time in layer_total_cuda_times]

        # we build a list with all the summaries of the leaf layers
        layers_summary = [summary for summary in self._torchinfo.summary_list
                          if not summary.inner_layers]
        # we build all the LayerInfo objects
        for layer_index, layer_time in enumerate(layer_rel_cuda_times):
            # to get the names of the modules, we can iterate over the traces
            # since we already have leaves, we don't have to filter them
            # we have to access the torch.nn.Module and get its name with the
            # hidden function
            # we use an index to distinguish the different layers with same
            # main kernel
            layer_name = layer_traces[layer_index]

            # we get the required sizes for the layer
            layer_input_size = self._torchinfo_summary.

        self._layer_stats.update({name: cuda_time
                                 for name, cuda_time in zip(layer_names,
                                                        layer_rel_cuda_times)})

    def _get_torchprof_profiling(self):
        if self._torchprof_profiling is not None:
            return

        # we create the tensor to match the correct input size
        tensor = torch.randn(*self.input_size, device='cuda')

        # FIXME: find a way of reverting this settings to the previous values
        # FIXED: it uses more memory but it doesn't change the original model
        temp_model = copy.deepcopy(self.model).eval().to('cuda')

        # here we need the temporary model, otherwise the results are wrong
        with torchprof.Profile(temp_model, enabled=True, use_cuda=True, profile_memory=True) as prof:
            temp_model(tensor)
        self._torchprof_profiling = prof

    def _get_torchinfo_summary(self):
        if self._torchinfo_summary is not None:
            return
        # verbose is set to 0 to avoid printing the results
        self._torchinfo_summary = torchinfo.summary(self.model, self.input_size,
                                               verbose=0)

    def _get_onnx_summary(self):
        # TO COMPLETE
        raise NotImplementedError

        # we generate the onnx export of the torch module to import it into
        # onnx and get the layer operations

        # dummy input
        tensor = torch.randn(*self.input_size, device='cpu')
        # output buffer, bytes
        export_buffer = io.BytesIO()

        # FIXME: find a way of reverting this settings to the previous values
        self.model.eval()
        self.model.to('cpu')

        torch.onnx.export(self.model,
                          tensor,
                          export_buffer,
                          # this is to avoid exporting the weight values, as
                          # we only require the model structure
                          export_params=False,
                          )

        # we reset the buffer to read it with onnx
        export_buffer.flush()
        export_buffer.seek(0)

    def _get_onnx_profiling(self):
        # TO COMPLETE
        raise NotImplementedError

        options = onnxruntime.SessionOptions()
        # we enable profiling
        options.enable_profiling = True
        session = onnxruntime.InferenceSession(path_to_model, options)

        input_name = session.get_inputs()[0].name

        dummy = numpy.random.rand(1, 3, 224, 224).astype(dtype=numpy.float32)

        session.run(None, {input_name: dummy})

        # profile name can't be chosen, but we can delete it after we read it
        prof_file = session.end_profiling()

    @property
    def layer_stats(self) -> typing.OrderedDict[int, LayerInfo]:
        return copy.deepcopy(self._layer_stats)

    @property
    def total_execution_time(self):
        return self._total_execution_time

    @property
    def raw_torchinfo_summary(self):
        return copy.deepcopy(self._torchinfo_summary)

    @property
    def raw_torchprof_profiling(self):
        return copy.deepcopy(self._torchprof_profiling)

    @property
    def raw_onnx_profiling(self):
        return copy.deepcopy(self._onnx_profiling)
