# Fault Injection Framework

* It uses PyTorch
  * In PyTorch it is sufficient to access _modules and change a single module to a Sequential containing the Module and the fault-injection part
    * The fault injection module could be adapted to avoid backward passing and being activated/deactivated via a flag, reachable by iterating through the modules

## Requirements

* PyTorch 1.4+
* torchvision 0.4.2+
* pandas 1.0.1+
* plotly (optional, for graphs)
* jupyter (optional, for interactive notebooks)

A complete list of requirements can be found in ```conda_env_pytorch_1_5_GPU.yml``` using PyTorch 1.5. It should be compatible also with CPU-only installations, as long as CUDA is not installable on the system.

## Package

The main package containing all the functions is located under ```src```. All the sub-packages and sub-modules make use of relative imports: in this way, even changing the name of the main directory will not change the behaviour of the packages, even though it is not suggested in the Python guidelines. The drawback of using relative imports is that the sub-packages and sub-modules are not accessible from within the package structure, e.g. ```import datasets``` from within ```src/dnn``` will not work.
