# TODOs for Fault Injection Framework

* Split the files and give a proper structure
* Use global configuration (INI / yaml + command line arguments)
* Use a plugin-like system, with different fault injectors / samplers which can be selected
* Support for weight injection
  * Support different memory mapping for low-level fault injection
  * Implement sparse operations (natively supported by PyTorch)
