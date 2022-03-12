# enpheeph

```enpheeph - Neural Fault Injection Framework```

We provide a customizable Fault Injection Framework, capable of injecting faults at different levels and with different libraries/networks.

## Installation

Download a release from GitHub release or from the dev branch and use

```pip install .```

This will install the selected release.

## Development

### Conventions

1. We follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for commit messages
2. We use [Semantic Versioning](https://semver.org/) for versions

### Tools

1. pytest/tox for testing
    1. Ideally we want 50% of the tests to be unit tests, testing the behaviour of each piece of code on its own.
    2. Then another 30% should be about integration tests on putting together different parts of the code.
    3. Finally the last 20% should cover some end-to-end tests, covering the whole flow from one side to the other
