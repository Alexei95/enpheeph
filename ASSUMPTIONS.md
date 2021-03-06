# Assumptions

Here we list all the assumptions we made for working on this library:

1. We assume only one main operation (conv2d, relu, ...) per layer in a DNN model;
   1. In this way, we can simplify the profiling for gathering timings, by using torchprof only
   2. Moreover, this simplifies also the modeling for the kernels, as we can focus on few kernels (conv, relu, gemm, ...) instead of having to implement all of the operations like flatten, reshape, ... which are done under the hood by PyTorch, as they do not are a limiting factor given their limited execution time.
   3. This allows us to consider total time (function call + children) instead of using self-time, which is not directly available in the profiler interface and needs to be computed manually.
      1. However, self_cpu_time_total (and correspondingly also CUDA self-time) exists for each EventList in the trace
2. Profiling is done on the GPU, so it requires to have the GPU which is gonna be the target of the experiments
   1. This limitation can be overcome in the future by allowing relative profiling, e.g. how much each operation contributes in a relative way
      1. This possible solution has limitations, as different platforms could handle different operations in an opposite way, leading to non-representative characterization
