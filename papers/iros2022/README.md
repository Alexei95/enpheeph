# enpheeph - Presentation Paper at IROS2022

This is the first paper based on the work done for ```enpheeph```, and it will be submitted to IROS2022.

The idea is to compare the performance in terms of latency, memory and developer effort.

## Framework Comparisons

1. [TorchFI](https://github.com/bfgoldstein/torchfi)
    1. Description
        1. Works on PyTorch
        2. Uses Python 3.7
    2. Pros
        1. Easy and modern setup
        2. Easy fault-aware training
    3. Cons
        1. Reimplements the injectable modules
2. [TensorFI](https://github.com/DependableSystemsLab/TensorFI)
    1. Description
        1. Works on Tensorflow v1
        2. Uses Python 2.7
    2. Pros
        1.
    3. Cons
        1. It requires writing a custom function/module for executing a specific injection
3. [TensorFI2](https://github.com/DependableSystemsLab/TensorFI2)
    1. Description
        1. Tensorflow v2
        2. Python 3
    2. Pros
        1. "Well"-documented
        2. Easy to run with examples
        3. One can choose which bits to set to zero/flip
    3. Cons
        1. No choice of layers/weights to be injected, they are always random
        2. Only one bit injection at a time
        3. No support for GPU/TPU injection
        4. The injection logic is run at run-time, so it adds more overhead
4. [PyTorchFI](https://github.com/pytorchfi/pytorchfi)
    1. Description
        1. Most recent changes, mid-November 2021
    2. Pros
        1. Similar to enpheeph, it uses forward hooks
        2. A lot of checks for dimensionality to avoid out-of-bound injections
    3. Cons
        1. Tied to PyTorch
        2. Values can be randomized completely in a range, but there is no sub-tensor specialization, i.e. we cannot inject at bit level
        3. No hardware optimization for GPU/TPU
5. [InjectTF2](https://github.com/mbsa-tud/InjectTF2)
    1. Description
        1. Evolution of InjectTF for TensorFlow2
        2. Caches the layer outputs to execute injections faster
    2. Pros
        1. Use of a caching algorithm to avoid the repetition of the execution of the non-faulty layer if the input is the same
    3. Cons
        1. No easy customizability for the injections
        2. Tied to TensorFlow2


## Fault Injection Examples

### Tasks - Datasets

1. Image Classification
    1. CIFAR10

2. Spiking Video Classification
    1. DVS128Gesture

### Configurations

Check ```configs/experiments``` for different configuration scripts.

**NOTE**: ```deterministic=True``` does not work in Trainer configuration for pruned image classifiers, it requires the update to PyTorch 1.11 as [it is fixed in master](https://github.com/pytorch/pytorch/issues/68525)

## Installation

To reproduce the results

1. Create a new conda environment ```conda create --name enpheeph-iros2022``` from the main enpheeph folder
2. Activate the environment with ```conda activate enpheeph-iros2022``` and install the requirements with ```conda install python=3.9.7 cupy=9.6.0 cudatoolkit=10.2 gcc_linux-64 gxx_linux-64```
3. Then install the pip dependencies with ```pip install -r papers/iros2022/requirements.txt --upgrade --upgrade-strategy only-if-needed```
    1. This will also install the downloaded enpheeph version
    2. If you are running ```lightning-flash==0.5.2``` as in the requirements, you need to patch the following file from the lightning-flash distribution, ```flash/core/utilities/flash_cli.py```, by running ```patch -u -b /path/to/python/site-packages/flash/core/utilities/flash_cli.py -i papers/iros2022/flash_cli_dataset_fix_0.5.2.patch```
4. download/train the weights
5. run the experiments
