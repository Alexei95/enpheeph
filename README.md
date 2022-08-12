# enpheeph

``enpheeph - Neural Fault Injection Framework``

We provide a customizable Fault Injection Framework, capable of injecting faults at different levels and with different libraries/networks.

**NOTE**: As of 2022-07-31, the framework is barely functional, hence there are still no official releases of the software. However, we plan on improving the framework over the course of the upcoming months to reach a point where it is stable enough to be used by researchers.

## Installation

To install the currently work-in-progress version use:

```pip install -e .```

This will install the downloaded source in editable mode, hence it will be linked directly to the download location, rather than being a copy in the default site-packages folder in the Python distribution. **It is highly recommended to use a virtual environment when installing enpheeph**

**NOTE**: ``enpheeph` cannot be used without installing it, as the dependency checks and the imports are absolute, i.e. independent of where the actual folder is, but require the installation to be working.

**NOTE**: while ``enpheeph`` should automatically install all the requirements, a full list for the complete version can be found in ``papers/iros2022/requirements.txt``

## Development

### Conventions

1. We follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages
2. We use [Semantic Versioning](https://semver.org/) for versions

### Tools

Install all the dependencies with

```pip install -e .[dev]```

## Citation

``enpheeph`` has been used for the following publications:

1. ```enpheeph: A Fault Injection Framework for Spiking and Compressed Deep Neural Networks``` to appear at [IROS 2022](https://iros2022.org).
   1. A preprint version is available on [arXiv](https://arxiv.org/abs/2208.00328), and it is also present in the repository citation button.

<!-- ### Tools -->

<!-- 1. pytest/tox for testing -->
<!--     1. Ideally we want 50% of the tests to be unit tests, testing the behaviour of each piece of code on its own. -->
<!--     2. Then another 30% should be about integration tests on putting together different parts of the code. -->
<!--     3. Finally the last 20% should cover some end-to-end tests, covering the whole flow from one side to the other -->
