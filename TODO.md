# TODOs for Fault Injection Framework

* Have models with different configurations depending on dataset
* Implement logging
* Implement pruning (structured + unstructured)
* Implement compression of saved models using algorithms
* Use global configuration (INI / yaml + command line arguments)
* Implement torch.jit.script with torch.jit.fork/wait for CPU parallelism (PyTorch 1.6)
* Support for weight injection
  * Support different memory mapping for low-level fault injection
  * Implement sparse operations (natively supported by PyTorch)
* Check out PyProf for profiling (not working with PyTorch 1.6)
* Check out PyTorch 1.6
* Convert everything to PyTorch Lightning

## DONE TODOs

* Split the files and give a proper structure
* Check about imports (whether to use src)
  * They can be made into relative using from .package import module
* Improve file structure
  * Put models and datasets under dnn, together with pruning and training/testing
* Use a plugin-like system, with different fault injectors / samplers which can be selected
  * Implement a global structure containing the injectors, samplers, models (e.g. in __init__)
* Integrate samplers
