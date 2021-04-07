# Table of Contents
- [Table of Contents](#table-of-contents)
- [Assumptions](#assumptions)

# Assumptions

Here we list all the assumptions we made for working on this library:

1. We assume only one main operation (conv2d, relu, ...) per layer in a DNN model;
    1. In this way, we can simplify the profiling for gathering timings, by using torchprof only
    2. Moreover, this simplifies also the modeling for the kernels, as we can focus on few kernels (conv, relu, gemm, ...) instead of having to implement all of the operations like flatten, reshape, ... which are done under the hood by PyTorch, as they do not are a limiting factor given their limited execution time.
    3. This allows us to consider total time (function call + children) instead of using self-time, which is not directly available in the profiler interface and needs to be computed manually.
        1. However, self_cpu_time_total (and correspondingly also CUDA self-time) exists for each EventList in the trace
2. Profiling is done on the GPU, so it requires to have the GPU which is gonna be the target of the experiments
    1. This limitation can be overcome in the future by allowing relative profiling, e.g. how much each operation contributes in a relative way
        1. This possible solution has limitations, as different platforms could handle different operations in a different way, leading to non-representative characterization
    2. Profiling a single layer compared to the result on the same layer when profiling the whole model are comparable, with no definite difference
3. We assume the hardware model provides the following pieces of information
    1. A list of elements, to describe the hierarchical structure of the GPU, each element is a dict which contains
        1. The Enum value describing the component
        2. The maximum number of elements in the GPU
        3. The Enum value corresponding to the direct parent
        4. A list of Enum values for sub-components
    2. A 2D array containing a map with all the component flags
        1. Each array element represent a unit of area and contains a list of component types
            1. Each element in the list contains
                1. A sequential number representing the sequential id of the component
                2. The Enum value of the component
            2. This is useful to map and schedule the kernels and threads on the components
4. The hardware model for the different elements respect the following conditions
    1. Kernels
        1. Convolution
            1. There is 1 thread for each output pixel
                1. This condition may be different depending from the implementation, in future it may be adapted to the different hardwares/SDK implementations
    2. Execution time is considered to be as if all threads run in parallel with no memory locks, therefore each thread will run in this much time in parallel with the others, not rising any memory conflicts or misses
    3. The components are numbered in increasing order from ```0``` to ```n_components - 1```
    4. All the target operators are available when starting an inference run
