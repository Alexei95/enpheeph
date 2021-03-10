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
