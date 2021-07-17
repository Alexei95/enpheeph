# Fault Injection Framework

## Installation

Use

```
pip -r requirements.txt
```

to install the package requirements for running the framework.

Extra requirements can be found in the folder ```requirements```, depending on the specific application which need to be used.

### Possible Issues

1. If installing ```tonic``` in a ```conda``` environment, pay attention to install also the GCC/G++ compilers from ```conda``` (e.g. ```gcc_linux-64``` and ```gxx_linux-64```, eventually also ```gfortran_linux-64``` if needed later on), as there are some ABI compatibility issues when building a ```tonic``` dependency, ```loris```, using a compiler which is external to the ```conda``` environment.
2. Pay attention to the CUDA libraries being used if installing both ```PyTorch``` and ```CUPy``` using ```conda```. The best solution is to install a compatible CUDA toolkit from ```conda```, and then install ```PyTorch``` and ```CUPy``` using the compatible versions from ```PyPI```, using ```pip```.

## Tests

Tests are run using ```pytest```.

To install all the dependencies just run

```
pip -r requirements/test.txt
```

This will install all the basic dependencies together with the extra ones for running the tests.

To run the tests, execute in the main project directory

```
python -m pytest
```
