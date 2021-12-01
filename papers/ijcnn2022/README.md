# enpheeph - Presentation Paper at IJCNN2022

This is the first paper based on the work done for ```enpheeph```, and it will be submitted to IJCNN2022.

The idea is to compare the performance in terms of latency, memory and developer effort.

## Framework Comparisons

1. TensorFI/BinFI by Pattabiraman
    1. Works on TensorFlow
    2. Uses Python 2.7
    3. It requires writing a custom function
2. TorchFI
    1. Works on PyTorch
    2. Uses Python 3.7

## Fault Injection Examples

### Tasks - Datasets

1. Semantic Segmentation, using PyTorch Lightning Flash
    1. CARLA Driving Dataset
2. Image Classification
    1. CIFAR10

## Installation

To reproduce the results

1. create conda environment
2. install the requirements
    1. patch flash cli (only if it has not been patched already in a future version)
    2. create the config files by running the .sh files in configs/experiments
3. install enpheeph (specify version/git commit)
4. download/train the weights
5. run the experiments
