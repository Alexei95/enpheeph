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


# this class simply works as a wrapper for the summaries required for computing
# the different parts of the model
# we mostly need operation-wise execution time (which can be relaxed to
# layer-wise, assuming one main operation per layer)
@dataclasses.dataclass(init=True)
class Summary(object):
    model: torch.nn.Module
    # input size must have the first element as batch size, which can be also 1
    input_size: typing.Tuple[int, ...]

    _torchinfo_summary = dataclasses.field(init=False, default=None)
    _torchprof_profiling = dataclasses.field(init=False, default=None)
    _onnx_profiling = dataclasses.field(init=False, default=None)
    _onnx_summary = dataclasses.field(init=False, default=None)
    _layer_stats = dataclasses.field(init=False,
                                     default_factory=collections.OrderedDict)

    def __post_init__(self):
        # we gather the layers info
        self._get_torchinfo_summary()
        #self._get_onnx_summary()

        # we gather the execution times
        self._get_torchprof_profiling()
        #self._get_onnx_profiling()

        # we populate the layer stats
        self._populate_layer_stats()

    def _populate_layer_stats(self):
        # we check if it has been already updated
        if self._layer_stats:
            return

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
        # FIXME: this works only with CUDA for now, must be adapted for CPU later
        layer_total_cuda_times = [sum(e.cuda_time for e in event_list)
                                  for event_list in layer_event_lists]

        # to get the names of the modules, we can iterate over the traces
        # since we already have leaves, we don't have to filter them
        # we have to access the torch.nn.Module and get its name with the
        # hidden function
        layer_names = [t.module._get_name() for t in layer_traces]

        self._layer_stats.update({name: cuda_time
                                 for name, cuda_time in zip(layer_names,
                                                    layer_total_cuda_times)})

    def _get_torchprof_profiling(self):
        if self._torchprof_profiling is not None:
            return

        # we create the tensor to match the correct input size
        tensor = torch.randn(*self.input_size, device='cuda')

        # FIXME: find a way of reverting this settings to the previous values
        # FIXED: it uses more memory but it doesn't change the original model
        temp_model = copy.deepcopy(self.model).eval().to('cuda')

        with torchprof.Profile(self.model, enabled=True, use_cuda=True, profile_memory=True) as prof:
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
    def raw_torchinfo_summary(self):
        return copy.deepcopy(self._torchinfo_summary)

    @property
    def raw_torchprof_profiling(self):
        return copy.deepcopy(self._torchprof_profiling)

    @property
    def raw_onnx_profiling(self):
        return copy.deepcopy(self._onnx_profiling)
