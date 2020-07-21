# Fault Injection Framework

* It uses PyTorch
  * It can be extended to use ONNX for model transfering
  * In PyTorch it is sufficient to access _modules and change a single module to a Sequential containing the Module and the fault-injection part
    * The fault injection module could be adapted to avoid backward passing and being activated/deactivated via a flag, reachable by iterating through the modules

## Requirements

* PyTorch 1.3.1+
* pandas 1.0.1+
* plotly (optional, for graphs)

A complete list of requirements can be found in ```conda_env_GPU.yml``` and ```conda_env_CPU.yml```, for GPU and CPU respectively.
