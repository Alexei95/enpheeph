# TODOs for Fault Injection Framework

* Have models with different configurations depending on dataset
* Check interdependency of modules
* Implement logging
* Implement pruning (structured + unstructured)
* Implement compression of saved models using algorithms
* Use global configuration (INI / yaml + command line arguments)
* Implement torch.jit.script with torch.jit.fork/wait for CPU parallelism (PyTorch 1.6)
  * While the injection is GPU-specific, it can be run on any capable device
* Support for weight injection
  * Support different memory mapping for low-level fault injection
  * Implement sparse operations (natively supported by PyTorch)
* Check out PyProf for profiling (not working with PyTorch 1.6)
* Check out PyTorch 1.6
* Convert everything to PyTorch Lightning (waiting for 0.9 release for DataModule)
  * Use seed_everything and deterministic flag in Trainer
  * Use argument parser together with Trainer
    * Check out hyperparameter saving
    * Check out model specific arguments
  * Use callbacks (model checkpointing, early stopping, ...)
    * Implement model compression (lzma) after model checkpointing
    * Check out model / trainer loading / saving
  * Implement a wrapper class for using external modules and datasets
  * Check support for colab TPUs for running experiments (ImageNet and similar)
* Pretrain models and upload them using Torch Hub
* Check Cross-Validation for averages
* Improve data classes, using the module dataclasses
* For now we use torch-summary for the summaries, but we can switch to PyTorch-Lightning when the support for FLOPS and memory size is available
* Add license info in all files
* Change discovery of models to use __init_subclass__ in DNN (register subclasses)
  * No need to update gather_objects, but we can skip using the values
* Implement a pipeline for testing the injections: the original module has stats collected on the ranges or the exact values of each layer/buffer, and then the module can be split, to be run in parallel for each layer to compare the results for possible faults
  * Another possibility is to model how to affect the outputs (ultimate goal of the fault injection campaign)
* Implement statistics for each layer/element

## DONE TODOs

* Split the files and give a proper structure
* Check about imports (whether to use src)
  * They can be made into relative using from .package import module
* Improve file structure
  * Put models and datasets under dnn, together with pruning and training/testing
* Use a plugin-like system, with different fault injectors / samplers which can be selected
  * Implement a global structure containing the injectors, samplers, models (e.g. in __init__)
* Integrate samplers
* Use MIT for license
