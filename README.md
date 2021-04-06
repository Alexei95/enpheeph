# Fault Injection Framework

We implement this framework to easily support fault injection campaigns, both for hardware-aware (GPU, ASIC) and for standard faults (MonteCarlo).

However, at the time of writing, only hardware-aware faults are supported, with normal faults and customizations to come in the future.

The focus is on CNNs on many datasets (MNIST, CIFAR10), with possible future implementations covering SNNs.

## Requirements

- Python 3.7+
    - Many algorithms are based around ordered dicts, which are natively supported in Python 3.7 and onwards
- PyTorch 1.4+
- PyTorch Lightning 1.0+

## Package
