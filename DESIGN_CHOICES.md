# Design Choices

Here we analyze the different design choices that have been made when building the different aspects of the application/framework.

# Table of Contents

- [Design Choices](#design-choices)
- [Table of Contents](#table-of-contents)
- [Model summary for fault injection](#model-summary-for-fault-injection)
    - [2021/03/10](#20210310)
    - [2021/03/26](#20210326)
- [Hardware Model](#hardware-model)
    - [2021/03/07](#20210307)
    - [2021/03/08](#20210308)
    - [2021/03/09](#20210309)
    - [2021/03/11](#20210311)
        - [GPU Characteristic and Structure](#gpu-characteristic-and-structure)
        - [GPU Internal Working and API](#gpu-internal-working-and-api)
    - [2021/03/12](#20210312)
        - [CUDA C++ Programming Guide Analysis](#cuda-c-programming-guide-analysis)
            - [**Kernel, Thread, Block and Grid**](#kernel-thread-block-and-grid)
            - [**Memory**](#memory)
            - [**Hardware Implementation**](#hardware-implementation)
            - [**Kernel/Thread Limits**](#kernelthread-limits)
        - [cuDNN Developer Manual Analysis](#cudnn-developer-manual-analysis)
            - [Memory Layout](#memory-layout)
    - [2021/03/15](#20210315)
    - [2021/03/16](#20210316)
        - [Warp Scheduling and Thread-CUDA Core Mapping](#warp-scheduling-and-thread-cuda-core-mapping)
    - [2021/03/17](#20210317)
        - [Hardware Model Decisions](#hardware-model-decisions)
        - [Hardware Model Implementation](#hardware-model-implementation)
    - [2021/03/18](#20210318)
        - [Hardware Model Implementation](#hardware-model-implementation-1)
    - [2021/03/19](#20210319)
        - [Hardware Model Implementation](#hardware-model-implementation-2)
    - [2021/03/22](#20210322)
        - [Hardware Model Implementation](#hardware-model-implementation-3)
    - [2021/03/23](#20210323)
        - [Hardware Model Implementation](#hardware-model-implementation-4)
    - [2021/03/26](#20210326-1)
        - [GPU API](#gpu-api)
        - [Hardware Model Implementation](#hardware-model-implementation-5)
    - [2021/03/28](#20210328)
        - [Hardware Model Implementation](#hardware-model-implementation-6)
    - [2021/03/29](#20210329)
        - [Hardware Model Implementation](#hardware-model-implementation-7)
    - [2021/03/31](#20210331)
        - [Hardware Model Implementation](#hardware-model-implementation-8)
    - [2021/04/01](#20210401)
        - [Summary Implementation](#summary-implementation)
    - [2021/04/05](#20210405)
        - [Hardware Model Implementation](#hardware-model-implementation-9)
    - [2021/04/06](#20210406)
        - [Hardware Model Implementation](#hardware-model-implementation-10)
    - [2021/04/07](#20210407)
        - [Hardware Model Implementation](#hardware-model-implementation-11)
    - [2021/04/08](#20210408)
        - [Hardware Model Implementation](#hardware-model-implementation-12)
    - [2021/04/13](#20210413)
        - [Summary Implementation](#summary-implementation-1)
        - [Hardware Model Implementation](#hardware-model-implementation-13)
    - [2021/03/14](#20210314)
        - [Hardware Model Implementation](#hardware-model-implementation-14)

# Model summary for fault injection

There are different ways of implementing the summary, depending on the required information:

1. Execution time -> to split the timings for the different operations
2. (Memory usage)
3. Used kernels -> together with input/output shapes

Overall, these are the main ways of getting parts of this list:

- onnx, using the structure of the model.graph (which acts as a list) to read the different elements, covering the kernels and their shapes (3)
- [torch.profiler](https://pytorch.org/docs/master/profiler.html) (for version 1.8, for 1.7.1 (2021/03/03) and before refer to [torch.autograd.profiler](https://pytorch.org/docs/master/autograd.html#profiler)) for getting execution time, memory usage and used kernels (1, 2 and 3)
    - However, the used kernels cover also many other "hidden" operations, like reshape, stride, ... which may make modeling all kernels much more difficult
    - This can be useful at a later time, when all the kernels can be easily mapped to a model
    - For now the focus should be on getting the basic kernels modeled (conv, relu, gemm)
    - A possible drawback is the estimation of the total time, which is not precise to the microsecond, by using "ts" from the chrome trace via json, but nonetheless it can provide a good estimate the bigger the model, as the relative error will get smaller given the longer runtime
- [torchprof](https://github.com/awwong1/torchprof), to get more detailed info on a layer-wise manner (1, 2 and 3), based on torch profiler, it allows for the same results but having also layer-wise analysis and number
- [torchinfo](https://github.com/tyleryep/torchinfo), which does not provide run-time numbers, but it approximates MACs, model size, and provides a list of layers/operations (2, 3)

Currently, the choice falls on torchprof for gaining info on the layer-wise execution time and run-time memory usage, together with onnx for knowing the actual high-level operations of each layer.

After an initial implementation, the best solution falls towards using onnx and onnxruntime for the used kernels together with their execution time: onnx has a very simple but expressive representation of fundamental kernels, while onnxruntime provide a compatible interface for profiling each kernel. The only drawback is that each layer may cover multiple layers, making it difficult to actually inject the faults.

Unfortunately, onnx and onnxruntime do not support sparse weight, so that would limit the usage of such info in our system. Hence, we fell back to torchinfo and torchprof. However, torchprof provides also some info regarding the leaf layers, that is the ones without children, which can be obtained also from torchinfo, by checking the property ```inner_layers``` to be empty for each ```LayerInfo``` object in ```summary_list```. This information can be more easily gathered through torchprof, because each trace for each layer has a corresponding object, with its path and attribute for determining whether it is a leaf. Once determined the leaves, the events can be traced but covering the whole sub-events. This list must be summed over all the sub-events (reshape, flatten, conv, ...) to obtain the actual layer time. This can be done over 4 variables, two for CPU and two for CUDA, two covering self-time and two covering total time. Self-time is the time of the sub-event without the child events, while total time includes also the children time.

A possible solution could be to use ```top_level_events_only``` in ```torch.autograd.profiler``` (which will become ```torch.profiler```), but this will be available at the time of writing in a future PyTorch release (1.8, done on 2021/03/04, currently on 1.7.1). UPDATE 2021/03/06, PyTorch 1.8, ```top_level_events_only``` still shows only the kernels, which can be useful but it is different from what we need (layer execution time).

Self-time is not directly available via the FunctionEvents in torchprof, and even using the raw interface to the torch profiler does not provide direct access. Therefore, the only way is recursively accessing the ```cpu_children``` attribute in each FunctionEvent, and subtract the children time from the parent's, to get the self-time. However, accessing each trace, the EventList itself, containing all of the FunctionEvents, contains also CPU and CUDA total self-time, and hence it can be used for implementation.

## 2021/03/10

Torchprof and torchinfo summaries are not directly matchable, as they may use different copies of the model, impossible to give equality checks across different modules.

A solution may be to go through the summaries and run the profiling on each sub-layer. However, this may lead to inconclusive results, as a run on sub-each sub-layer may be different from a consecutive run over all the layers at the same time. More testing is required.

Testing the profiling, using only one layer, we obtained 84.15672599999985 ms as measurement, while using the whole model we got 85.045844 ms. Therefore, given the variability of profiling, the results are close enough to be interchanged.

The point is to compute the summary of the model, and profile each leaf layer independently starting from the summarised layers. In this way we are sure the data belong to the correct layer, while obtaining very accurate results.

Final implementation has layer summaries associated with raw profiling results, and a specialized list with all the required info for each layer.

## 2021/03/26

PyTorch has released version 1.8.1 with the new profiler, which provides better interface and customizability. However, it does not provide any automatic grouping as in torchprof, therefore we will stick to torchprof to maintain automatic grouping capabilities, while they update the interface to support new functionalities.

# Hardware Model

## 2021/03/07

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

## 2021/03/08

We may need a higher-level model of the GPU, as there are some high-level properties, such as control and memory sections, together with shared instruction info, scheduling, interconnections. It can be modeled similarly to a single kernel, being split in subcomponents (Interconnections, Control, Memory, ...). A possibility is to have place-holder slots for Kernels, which are filled based on the time.

Using kernel slots, we need to limit the processing sizes, so that we can map the correct number of operations, i.e. a conv2d with 100x100 to cover requires more operations than a 10x10, and this must be taken into account.

## 2021/03/09

Kernel slots should be used to fill the operations, however the number of operations and the timings are not to be considered exact as they are, given that we only need an estimate of the associations.

The overall objective is to have a precise location for when the fault occurs, in terms of weight, input and output.

Each slot, given the input/output capabilities of the kernel, is used to address the number of repetitions of the operation.

Currently, Summary provides a dict containing the info for layer and execution time. However, it can be improved by using only the layer index and adding info such as input/output size, execution time, name of the kernel.

## 2021/03/11

### GPU Characteristic and Structure

We will base our model on the GA102 GPU, which is the biggest available version of the [Ampere Architecture by Nvidia](https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.pdf). This chip is used in the Nvidia RTX A6000, RTX A40 and RTX 3090 FE (actually the RTX 3090 has only 41 TPCs, instead of the theoretical maximum of 42).

Each GPU is split into Graphic Processing Clusters (GPCs), at the highest hardware level. It contains:

- Dedicated Raster engine
- 2 Raster Operator (ROP) partitions, each containing 8 ROPs
- 6 Texture Processing Clusters (TPCs), each containing:
    - 2 Streaming Multiprocessors (SMs), each composed of:
        - 128 CUDA cores
            - 64 FP32 datapaths, 16 per processing block
            - 64 FP32/INT32 datapaths, 16 per processing block
        - 1 2nd-gen Ray Tracing Core
        - 128 KB Shared L1/Shared Memory (the amount can be configured)
        - 2 FP64 units, TFLOPS at 1/64th the TFLOPS of FP32 operations
        - 4 processing blocks, which split the previous resources, plus each of them containing:
            - 1 3rd-gen Tensor Cores
            - 64 KB (16,384 x 32bit) Register File
            - 1 Texture Units
            - L0 instruction cache
            - one warp scheduler
            - one dispatch unit (32 thread/clk)
    - 1 PolyMorph Engine

There are 12 32-bit memory controllers, covering a total of 384 bits. Each controller has 512 KB L2 cache, for a total of 6144 KB.

Total characteristics:

- 7 GPCs
- 112 ROPs
- 168 FP64 units
- 336 3rd-gen Tensor Cores
- 84 Ray-Tracing Cores
- 336 Texture Units
- 6144 KB L2 Cache
- 21504 KB Register File
- 628.44 mm2 Die Size
- 28.3 Billion Transistors

### GPU Internal Working and API

This part is important to understand the way the GPU processes the instructions and the data.

There are different libraries:

- CUDA is the generic C++ library used to run code on the Nvidia GPUs
- cuDNN is a library built on top of CUDA to define commonly used functions, such as convolutions, activation functions and other utilities, to provide a unified and optimized set of functions to all CUDA users interested in Machine Learning applications

Some documentation exists:

- [CUDA Documentation](https://docs.nvidia.com/cuda/), which contains all the documentation for understanding and developing CUDA-compatible code
    - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf), which provides documentation on the way the GPU works and interprets the instructions, together with all the bell and whistles required to implement optimized software to be run on GPUs
        - This document seems very useful as it describes the high-level internal operation of threads and warps, so that it is easy to understand where faults could occur
    - [Tuning CUDA Applications for Ampere](https://docs.nvidia.com/cuda/pdf/Ampere_Tuning_Guide.pdf), a guide containing extra info specific for Ampere GPUs
        - It contains extra info, which may be specific to Ampere, useful for having a good model
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/index.html), it contains all the documents for using cuDNN, together with some detailed explanations
    - [cuDNN Developer Manual](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Developer-Guide.pdf), which describes the operations done in the cuDNN library and their implementation
        - The developer manual seems useful for memory accessing patterns, as there are descriptions of data layouts, accesses and operations when executing this specific functions


## 2021/03/12

### [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf) Analysis

#### **Kernel, Thread, Block and Grid**

Kernels are C++ functions, which are executed N times in parallel by N different CUDA threads. Each thread is given a specific ID, which relates to an index in the corresponding _thread block_, which is up to 3-dimensional. The ID can be computed from the index, by using ```x + y * Dx + z * Dx * Dy```, with (Dx, Dy, Dz) being the dimensions of the thread block. Each kernel can be executed across thread blocks, to overcome the limit of 1024 threads per thread block, which are further organized in structures called _grids_. There are similar block IDs and indices as thread indices.

```cpp
// C is the destination matrix, A and B are N x N matrices
MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
```

#### **Memory**

Memory is local for each thread, it is shared across threads in the same block, while across grids and blocks it is required to access global memory.

#### **Hardware Implementation**

Instructions are pipelined in each CUDA core, they are issued and executed in order, so there is no speculative execution and no branch prediction.

Threads are handled in groups of 32, called warps. Each thread in a warp starts from the same programm address and state, but then they execute independently, with its own state, call stack and program counter, so they are free to branch.

Each block is split in warps by the warp scheduler, using increasing thread IDs, starting from 0.

The warp executes only common instructions, so the execution path must be the same for all the threads. If any of them branches, all the possible branches are executed in sequence, disabling the other threads until the warp reaches again a common point. For instructions on common memory, there are limits on the serialization if the instructions are non-atomic, if atomic they are all serialized by the warp but execute in non-deterministic order.

#### **Kernel/Thread Limits**

* Maximum Number of Threads per Thread Block: 1024


### [cuDNN Developer Manual](https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-Developer-Guide.pdf) Analysis

#### Memory Layout

There are different memory layouts which are used by the cuDNN library. N refers to the batch size, C to the channels (features), H to the height of the kernel and W to the width:

- **NCHW**, where all the elements are sequential in memory, first going through width, then height, finally channels and batches
- **NHWC**
- **NC/xHWx**, where all the channels are grouped into groups with _x_ elements, where all the first elements are taken from the first _x_ groups, then these channels are run through, to start with the following _x_ elements

## 2021/03/15

The previous description should be comprehensive enough to cover most of the design choices required at the beginning of the design phase.

## 2021/03/16

### Warp Scheduling and Thread-CUDA Core Mapping

There are few pieces of information regarding the way the warp scheduler works.

Therefore, we require extra sources to model the way the kernels are scheduled.

Here is a list of good references:

- [Thread Block on Wikipedia](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)), it explains how thread blocks are mapped to CUDA cores and SM
    - Here we have the description of a warp being 32 threads wide
    - Each SM can execute multiple threads, split into warps, but they must execute the same instruction
        - All the warps of a block are associated to the same SM, and new warps are not loaded if they cannot fit all inside the SM
        - For a warp to be executed, all of its threads must be ready to execute, otherwise we context switch to a different warp
            - This is generally due to to external data dependencies, because if the operands are local, they are directly available
    - Each SM executes only one warp
    - [Extra Reference](https://users.umiacs.umd.edu/~ramani/cmsc828e_gpusci/lecture9.pdf)
        - Here is a more detailed explanation, covering more numerical examples
        - Threads are assigned in block granularity
        - It depends on the block configuration
        - Only one warp can execute per SM at any given time
    - However, here we are still missing the association warp/CUDA core
    - [Extra Reference 2](https://indico.in2p3.fr/event/22228/contributions/87050/attachments/60191/81733/GDR_CUDA_introduction_vom_Bruch_October2020.pdf)
        - More details and comparisons across GPUs/brands
        - **Here we have the important notion that each thread runs in a single CUDA core**
        - Also that block size is multiple of warp size
        - We can synchronize for global memory across blocks, or with __synchthreads() inside each block, to create a wall for having all the kernels finishing their job
- [Medium Article about CUDA and GPU](https://jonathan-hui.medium.com/ai-chips-a100-gpu-with-nvidia-ampere-architecture-3034ed685e6e)
    - It explains in detail how the GPU works, with warps, threads and thread blocks
    - It covers some of the instructions done by each CUDA core
    - Inside each SM there are also Special Function Units (SFUs), to accelerate operations which cannot be easily done in standard FP32/INT32 cores
- [Extra Reference 2](https://www.cse.iitk.ac.in/users/swarnendu/courses/autumn2019-cs698l/GPU-CUDA.pdf)
    - More details on CUDA/GPU, with analysis on software perspective and optimization decisions
    - There are issues in accessing main memory for each thread in a warp if they want to access the same bank (the division of the main memory)
        - This forces the accesses to become sequential, losing performance
        - Otherwise they work in parallel

## 2021/03/17

### Hardware Model Decisions

Given all the previous information, we can make educated decisions on how to model the GPU:

- First of all, there will be the need for having a list of numbers, covering the overall technical specifications
    - This can be configured as a dataclass, so that it can be changed in the future without breaking compatibility, by simply providing as base class a certain set of properties
        - Examples
            - \# of SMs
            - \# of GPCs
            - \# of TPCs
            - \# of CUDA cores
        - These numbers can also be converted into relative ones, e.g. \# of SMs per TPC, and give one absolute number to convert them all
    - Then, we need to have some software-level info, like
        - \# of threads in a warp
        - This is useful for scheduling and knowing the timeline of the different kernels being run in the GPU
    - The GPU model should also contain a scheduling list of all the warps, together with their physical positioning for injecting the faults
- After the general model with the info, we need a base Kernel class, which covers all the info required to run a kernel
    - For example, it may contain the dimensions of the thread block, together with the number of warps
    - It should also cover some essential information on the memory and the operations which are run inside each CUDA core
- The interaction between the different parts is that the GPU model has a timeline of kernels, and the instantiated objects from the base class Kernel are used to populate the low-level timeline, which is then used to compare with physical particle interaction

### Hardware Model Implementation

To implement the actual hardware representation, we are developing

- A class Kernel, to contain the base info on each kernel
    - thread_block_size, the dimensions of the thread block which will execute the kernel, it must a tuple with dimensions, e.g. (24, 16, 8)
    - input_size, a tuple where each element is a tuple with the input dimensions of the kernel
    - output_size, similar to input_size, each element of the tuple has the output dimensions of the kernel
    - thread_input_size, same as input size, but for only a single thread
    - thread_output_size, same as output_size, but for a single thread
- An Enum, one for each GPU/brand
    - It defines all the possible components, as these are required to map the GPU structure
- A HardwareModel class, which contains the total number of components and the possible interactions, together with the scheduling of such operations
    - A hierarchy must be passsed, which should be loaded from a JSON file
        - This JSON file should be structured with concatenated lists, where each element is a list, with the Enum flag corresponding to the component, followed by the maximum number, and by a list of sub-components, example
            - ```{'component': {'__enum__': 'NvidiaGPUComponentEnum.StreamingMultiprocessor'}, 'max': 24, 'subcomponents': [...]}```

## 2021/03/18

### Hardware Model Implementation

As an update to the previous implementation, we decided to trade off the compact representation in the JSON for an easier representation to parse

We use a dict of components, having the enumeration value as key and the component as value, and each component, instead of having a list with directly the subcomponents, it contains a list with all their enumeration values, so that they can be addressed directly from the dict.

In addition, each component has also a parent, to allow back-tracking. This is useful when going from leaves to the root, e.g. when computing the number of total subcomponents. This parameter could be a list, but it does not make much sense as a single subcomponent cannot have multiple parents, as they need to have a certain hierarchy

## 2021/03/19

### Hardware Model Implementation

- Kernel output size must contain a single element, otherwise we may have trouble compiling the total size
    - Same for thread output size


## 2021/03/22

### Hardware Model Implementation

- Compute number of threads based on output size for kernel and for thread
    - We compute the number of threads from the number of elements of the output, divived by the number of elements processed by each thread

## 2021/03/23

### Hardware Model Implementation

## 2021/03/26

### GPU API

CUDA Streams are used to issue operations in-order on GPUs: all the instructions in a stream are issued in order, however instructions can be executed in parallel using multiple streams, if they do not block each other, e.g. a computation on data which has already been loaded.

### Hardware Model Implementation

We need to model each different kernel, and provide the number of threads and some more details on them.

As a starting point, we can implement the conv2d algorithm. In our case, we will use the [cuDNN API Reference manual focusing on the forward cuDNN convolution function call](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionForward). In this case, there are all the details regarding the way the function call works. However, there are some details missing:

- **There is no description of the actual algorithm implementations**: while different algorithms can be chosen (Winograd, GEMM, direct, FTT, ...), there is no direct description of the implemented algorithm. This limits the modelling capabilities, as we cannot know how many threads are required.

## 2021/03/28

### Hardware Model Implementation

Several references are available describing how to choose the number of threads for each CUDA operation. However, none of them represents the official cuDNN implementation, which is the one used also by PyTorch and most high-level Deep Learning libraries.

Therefore, we will analyze the different implementations and choose the optimal one.

- [CUDA Separable Convolution by NVIDIA (2007)](https://developer.download.nvidia.com/assets/cuda/files/convolutionSeparable.pdf): in this document there is an old explanation regarding how to implement separable kernels for convolutions, that is kernels which can be split into two-dimensional operations instead of going over all the elements at once. Here there is also an explanation regarding how to choose the number of threads, and the final choice is **1 thread per output pixel**.
    - [Lectures based on Separable Convolution](https://www.evl.uic.edu/sjames/cs525/final.html)
    - [GPU Programming Course](https://www.evl.uic.edu/aej/525/)
- [Convolutions in TVM, end-to-end deep learning library by Apache](https://tvm.apache.org/docs/tutorials/optimize/opt_conv_cuda.html): here we get a similar explanation regarding threads, where the number of threads depend on the dimensions of the kernel window, in this particular case, **each thread goes over a 4x4 grid**, which is strided by 4 in both directions to fill a 8x8 grid for the block.

Final decision is to assume **1 thread per output pixel**, and obtaining the total runtime from that.

## 2021/03/29

### Hardware Model Implementation

We added extra fields for the Kernel class, being the base class from which we can derive all the kernels.

Now, we have input, weight, bias and output sizes, each of them being a single tuple contatining all the dimensions. If one of them is not needed, it will be left as an empty tuple. Similarly, we have the same sizes for each thread running the kernel, together with some constants (THREADS_PER_WARP, defining how many threads are in a warp) and some run-time properties (n_threads, n_warps).

For scheduling, there will be a dict using the CUDA core identifiers (since we run **each thread in a single CUDA core**)

## 2021/03/31

### Hardware Model Implementation

To consider the scheduling issue, we have to include a new ThreadDescriptor class, which contains start, stop and length times, as well as thread "id", which is a sequential integer for the kernel, a link to the parent kernel and to the CUDA core id.

## 2021/04/01

### Summary Implementation

Updated the total execution time to be in seconds instead of microseconds, to have a standard value. Relative conversion stays the same as it is in microseconds all the time.

## 2021/04/05

### Hardware Model Implementation

For scheduling properly, and maximizing the utilization of the GPU, we need to select multiple subsets of the free operators, until all of them are exhausted.

## 2021/04/06

### Hardware Model Implementation

We don't need object-wide variables for free and scheduled operators, as these dicts will be handled on a per-run basis.

## 2021/04/07

### Hardware Model Implementation

We standardize the way we count the number of components, being it expressed in a per-parent basis, using an explicit variable for mentioning it.

Removed the target IDs argument when scheduling, now we assume that they are all contiguously numbered from ```0``` to ```n_components - 1```.

## 2021/04/08

### Hardware Model Implementation

There has been a slight redesign in the JSON import/export, now the model uses a registering process for saving the methods for encoding/decoding the corresponding classes. It has been implemented using a decorator, to make it easier, together with base classes implementing the methods for Enum and standard \_\_dict\_\_-based classes.

This update will be very useful in the future when adding more Enums and classes, however it required a change from typing.NamedTuple to dataclasses.dataclass, as the former has issues with inheritance related to the way the fields and field defaults are used in the metaclass.

Also, there is now a property returning the hierarchy JSON, as the internal representation is not a list of components as for the input hierarchy JSON, but instead it is a dict matching the enum as key with the component as value, for access efficiency.

All the properties have also been converted to functools.cached_property, given the high cost in terms of computation time, which may be useful in future for repetitive access.


## 2021/04/13

In the last few days there has been a deep redesign of many sections of the DNN and hardware models, which are covered in the following sections.

This redesign stems from issues found when trying to schedule operations, as we need kernel numbers which must be deduced from the layer which we are trying to model. Hence, there is the need for extra information, such as an Enum type for layer information, which will be extended with different main kernels depending on situations, as well as a rewriting of the Enum-to-JSON conversion, which is now universal as long a encoder/decoder are chosen.

While this information is important, it is not necessary for the basic functioning of the model, as we only require the number of threads used for each output element. However, this information **MUST** be defined in the future to take into account memory/dataflow locks in the GPU, which are ever so important for big models.

### Summary Implementation

Here the redesign covers mostly the type of information we are saving in the LayerInfo class.

We have added a new Enum type to cover for the different types of main kernel in each layer. This enum list will be expanded over time, to cover for new kernel types as well. A new property is the original execution time, obtained from the GPU profiling. Also, the LayerInfo class is now a dataclass, so that it is more flexible in handling arguments and extra methods. Currently it supports all the layers defined in AlexNet. In addition it provides a ```from_string``` class method, which is used to convert the string obtained from the layer profiling to a value from the Enum.

Regarding new LayerInfo methods, we have added a parse_representation static method, which is used to get the arguments of a call from its string representation. An example in this sense can be ```Conv2d(32, 64, kernel=(3, 3), stride=(1, 1))```, which would return ```{'__args__': [32, 64], 'kernel': (3, 3), 'stride': (1, 1)}```. This function is used in the ```__post_init__``` method to initialize a new property called parsed_representation. This property is used as new property extra_args in Kernel, which is parsed further to determine the sizes for thread input, weight and bias.

### Hardware Model Implementation

Here there have been the most changes: the whole JSON conversion system has been completely redesigned, leading to a class implementing a custom encoder and decoder mechanism. The encoder is a method, as it requires a JSONEncoder inheritance to work, while the decoder is a a classmethod.

While it could have been a static method as well, we have chosen the class method to implement a registering system for the encodable objects. In this way, each object can register its own encoder/decoder which will then be saved in a dictionary using the class name as the key, and be called when needed for JSON conversion. We have also implemented a decorator for automatically registering classes which implement a ```to_json``` and a ```from_json``` functions.

The JSON representation revolves around having a key in the output JSON dict called ```___cls___```, which contains the name of the class, together with an expanded dict representation of the object provided by the object encoder itself. For decoding, a similar procedure is followed, by calling the corresponding decoder with the dict-encoded object representation.

For helping in expanding the list of compatible objects, we have also defined a new abstract base class, defining the ```to_json``` and ```from_json``` methods as abstract ones, and therefore forcing the child classes to implement them. Some basic classes, like enumerations, dataclasses or dict-based classes are covered in two custom classes, ```JSONSerializableDictClass``` and ```JSONSerializableEnum```, which can be easily subclassed. These base classes help in accelerating future development, as it is as easy as inherit from these classes and use the registering decorator to be able to serialize a JSON object.

Other improvements cover the conversion of NamedTuple to dataclasses, to be able to implement methods for JSON conversion, as well as reorganizing the sample hierarchy and map. Also, small adjustments to names of some parameters and variable names have been made.

## 2021/03/14

### Hardware Model Implementation
