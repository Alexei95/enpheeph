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
