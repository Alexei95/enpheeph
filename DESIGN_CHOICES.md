# Design Choices

Here we analyze the different design choices that have been made

## Model summary for fault injection

There are different ways of implementing the summary, depending on the required information:

1. Execution time -> to split the timings for the different operations
2. (Memory usage)
3. Used kernels -> together with input/output shapes

Overall, these are the main ways of getting parts of this list:

* onnx, using the structure of the model.graph (which acts as a list) to read the different elements, covering the kernels and their shapes (3)
* [torch.profiler](https://pytorch.org/docs/master/profiler.html) (for version 1.8, for 1.7.1 (2021/03/03) and before refer to [torch.autograd.profiler](https://pytorch.org/docs/master/autograd.html#profiler)) for getting execution time, memory usage and used kernels (1, 2 and 3)
  * However, the used kernels cover also many other "hidden" operations, like reshape, stride, ... which may make modeling all kernels much more difficult
  * This can be useful at a later time, when all the kernels can be easily mapped to a model
  * For now the focus should be on getting the basic kernels modeled (conv, relu, gemm)
  * A possible drawback is the estimation of the total time, which is not precise to the microsecond, by using "ts" from the chrome trace via json, but nonetheless it can provide a good estimate the bigger the model, as the relative error will get smaller given the longer runtime
* [torchprof](https://github.com/awwong1/torchprof), to get more detailed info on a layer-wise manner (1, 2 and 3), based on torch profiler, it allows for the same results but having also layer-wise analysis and number
* [torchinfo](https://github.com/tyleryep/torchinfo), which does not provide run-time numbers, but it approximates MACs, model size, and provides a list of layers/operations (2, 3)

Currently, the choice falls on torchprof for gaining info on the layer-wise execution time and run-time memory usage, together with onnx for knowing the actual high-level operations of each layer.

After an initial implementation, the best solution falls towards using onnx and onnxruntime for the used kernels together with their execution time: onnx has a very simple but expressive representation of fundamental kernels, while onnxruntime provide a compatible interface for profiling each kernel. The only drawback is that each layer may cover multiple layers, making it difficult to actually inject the faults.

Unfortunately, onnx and onnxruntime do not support sparse weight, so that would limit the usage of such info in our system. Hence, we fell back to torchinfo and torchprof. However, torchprof provides also some info regarding the leaf layers, that is the ones without children, which can be obtained also from torchinfo, by checking the property ```inner_layers``` to be empty for each ```LayerInfo``` object in ```summary_list```. This information can be more easily gathered through torchprof, because each trace for each layer has a corresponding object, with its path and attribute for determining whether it is a leaf. Once determined the leaves, the events can be traced but covering the whole sub-events. This list must be summed over all the sub-events (reshape, flatten, conv, ...) to obtain the actual layer time. This can be done over 4 variables, two for CPU and two for CUDA, two covering self-time and two covering total time. Self-time is the time of the sub-event without the child events, while total time includes also the children time.

A possible solution could be to use ```top_level_events_only``` in ```torch.autograd.profiler``` (which will become ```torch.profiler```), but this will be available at the time of writing in a future PyTorch release (1.8, done on 2021/03/04, currently on 1.7.1). UPDATE 2021/03/06, PyTorch 1.8, ```top_level_events_only``` still shows only the kernels, which can be useful but it is different from what we need (layer execution time).

Self-time is not directly available via the FunctionEvents in torchprof, and even using the raw interface to the torch profiler does not provide direct access. Therefore, the only way is recursively accessing the ```cpu_children``` attribute in each FunctionEvent, and subtract the children time from the parent's, to get the self-time. However, accessing each trace, the EventList itself, containing all of the FunctionEvents, contains also CPU and CUDA total self-time, and hence it can be used for implementation.

### 2021/03/10

Torchprof and torchinfo summaries are not directly matchable, as they may use different copies of the model, impossible to give equality checks across different modules.

A solution may be to go through the summaries and run the profiling on each sub-layer. However, this may lead to inconclusive results, as a run on sub-each sub-layer may be different from a consecutive run over all the layers at the same time. More testing is required.

Testing the profiling, using only one layer, we obtained 84.15672599999985 ms as measurement, while using the whole model we got 85.045844 ms. Therefore, given the variability of profiling, the results are close enough to be interchanged.

The point is to compute the summary of the model, and profile each leaf layer independently starting from the summarised layers. In this way we are sure the data belong to the correct layer, while obtaining very accurate results.

Final implementation has layer summaries associated with raw profiling results, and a specialized list with all the required info for each layer.

## Hardware Model

### 2021/03/07

The current idea for hardware model is to have a set of classes, maybe with a common base class (Kernel), which are used to model the behaviour of a kernel when a particle hits.

When a particle hits, we need to consider the underlying technology (silicon or whatever), which generates a fault at a bit level given the proper interaction. This interaction can be summed up as a probability of generating a fault from a particle strike. The probablity can be derived from other works or by doing Monte-Carlo simulations.

Once a fault exists in the system, it needs to propagate. For propagation there are different options:

1. Detailed model of each bit/interconnection used in the kernel, which therefore could follow a precise physical model for particle interaction
   1. It could be implemented using a grid
      1. A grid with such small dimension would be too difficult to actually use, as it would incur a huge overhead
   2. Use an inverse map, where each element has its own boundary limits, and we check which element the particle hits by going through the list
      1. This solution may work, but it requires definitions for many elements, such as Interconnection, Computing, StaticMemory, ...

After propagating the fault from the particle strike to the kernel, it is a matter of modeling the kernel itself. Also here, there are different choices:

1. Modeling probabilistically the effect of a fault on any element to the output of the kernel
   1. It could use estimated values
      1. Taken from other Monte Carlo papers
      2. Taken from real-world simulations
   2. We could simulate some kernels using other injection tools ([NVBitFI](https://github.com/NVlabs/nvbitfi), on a side note, it could be useful for integrating ISA-level fault injections on PyTorch code, by converting PyTorch to TorchScript (C++) and using NVBitFI on it), however it could be time-expensive, better suited for a journal version
2. Directly executing the script to propagate the fault, which could be feasible on open-source GPUs, where the execution path is known, but it is an issue per se on closed-source GPUs.

### 2021/03/08

We may need a higher-level model of the GPU, as there are some high-level properties, such as control and memory sections, together with shared instruction info, scheduling, interconnections. It can be modeled similarly to a single kernel, being split in subcomponents (Interconnections, Control, Memory, ...). A possibility is to have place-holder slots for Kernels, which are filled based on the time.

Using kernel slots, we need to limit the processing sizes, so that we can map the correct number of operations, i.e. a conv2d with 100x100 to cover requires more operations than a 10x10, and this must be taken into account.

### 2021/03/09

Kernel slots should be used to fill the operations, however the number of operations and the timings are not to be considered exact as they are, given that we only need an estimate of the associations.

The overall objective is to have a precise location for when the fault occurs, in terms of weight, input and output.

Each slot, given the input/output capabilities of the kernel, is used to address the number of repetitions of the operation.

Currently, Summary provides a dict containing the info for layer and execution time. However, it can be improved by using only the layer index and adding info such as input/output size, execution time, name of the kernel.

### 2021/03/11

#### GPU Characteristic and Structure

We will base our model on the GA102 GPU, which is the biggest available version of the [Ampere Architecture by Nvidia](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf). This chip is used in the Nvidia RTX A6000, RTX A40 and RTX 3090 FE (actually the RTX 3090 has only 41 TPCs, instead of the theoretical maximum of 42).

Each GPU is split into Graphic Processing Clusters (GPCs), at the highest hardware level. It contains:
    * Dedicated Raster engine
    * 2 Raster Operator (ROP) partitions, each containing 8 ROPs
    * 6 Texture Processing Clusters (TPCs), each containing:
        * 2 Streaming Multiprocessors (SMs), each composed of:
            * 128 CUDA cores
                * 64 FP32 datapaths, 16 per processing block
                * 64 FP32/INT32 datapaths, 16 per processing block
            * 1 2nd-gen Ray Tracing Core
            * 128 KB Shared L1/Shared Memory (the amount can be configured)
            * 2 FP64 units, TFLOPS at 1/64th the TFLOPS of FP32 operations
            * 4 processing blocks, which split the previous resources, plus each of them containing:
                * 1 3rd-gen Tensor Cores
                * 64 KB (16,384 x 32bit) Register File
                * 1 Texture Units
                * L0 instruction cache
                * one warp scheduler
                * one dispatch unit (32 thread/clk)
        * 1 PolyMorph Engine

There are 12 32-bit memory controllers, covering a total of 384 bits. Each controller has 512 KB L2 cache, for a total of 6144 KB.

Each SM contains

Total characteristics:
    * 7 GPCs
    * 112 ROPs
    * 168 FP64 units
    * 336 3rd-gen Tensor Cores
    * 84 Ray-Tracing Cores
    * 336 Texture Units
    * 6144 KB L2 Cache
    * 21504 KB Register File
    * 628.44 mm2 Die Size
    * 28.3 Billion Transistors


#### GPU Internal Working and API

This part is important to understand the way the GPU processes the instructions and the data.

There are different libraries:
    * CUDA is the generic C++ library used to run code on the Nvidia GPUs
    * cuDNN is a library built on top of CUDA to define commonly used functions, such as convolutions, activation functions and other utilities, to provide a unified and optimized set of functions to all CUDA users interested in Machine Learning applications

Some documentation exists:
    * [CUDA Documentation](https://docs.nvidia.com/cuda/), which contains all the documentation for understanding and developing CUDA-compatible code
        * [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf), which provides documentation on the way the GPU works and interprets the instructions, together with all the bell and whistles required to implement optimized software to be run on GPUs
            * This document seems very useful as it describes the high-level internal operation of threads and warps, so that it is easy to understand where faults could occur
        * [Tuning CUDA Applications for Ampere](https://docs.nvidia.com/cuda/pdf/Ampere_Tuning_Guide.pdf), a guide containing extra info specific for Ampere GPUs
            * It contains extra info, which may be specific to Ampere, useful for having a good model
    * [cuDNN Developer Manual](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Developer-Guide.pdf), which describes the operations done in the
