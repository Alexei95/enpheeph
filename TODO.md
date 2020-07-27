# TODOs for Fault Injection Framework

* Integrate samplers
* Have models with different configurations depending on dataset
* Implement logging
* Implement pruning (structured + unstructured)
* Implement compression of saved models using algorithms
* Use global configuration (INI / yaml + command line arguments)
* Support for weight injection
  * Support different memory mapping for low-level fault injection
  * Implement sparse operations (natively supported by PyTorch)

# DONE TODOs

* Split the files and give a proper structure
* Check about imports (whether to use src)
  * They can be made into relative using from .package import module
* Improve file structure
  * Put models and datasets under dnn, together with pruning and training/testing
* Use a plugin-like system, with different fault injectors / samplers which can be selected
  * Implement a global structure containing the injectors, samplers, models (e.g. in __init__)
