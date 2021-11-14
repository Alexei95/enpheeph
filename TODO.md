# TODOs

1. #1 Add typing.Protocol implementation for ExperimentRun and Injection, so that they can be used as return type in StoragePluginABC
2. #2 Implement tests for all of the plugins and interfaces
3. #3 Fix all the CI/CD settings
    1. #3-1 Improve pre-commit and tool configurations
        1. #3-1-1 Use pytorch_lightning as example
4. #4 Update all the code to follow black and flake8
5. #5 Add documentation/examples on how to use the different injections
    1. #5-1 Add docstrings and more comments inside the code
    2. #5-2 Use Sphinx and reST for documents, follow pytorch_lightning for some examples
        1. #5-2-1 Docs can be hosted for free on readthedocs.io
6. #6 Add PEP-561 compliance (py.typed)
7. #7 Check for MANIFEST.inhttps://youtu.be/u6tafIJ6Z6c
8. {#8} Write proper ```__init__.py``` to import the classes from the sub-modules
    1. {#8-1} ```mkinit``` does that, run ```mkinit --recursive --black --lazy src/enpheeph -w```
        1. {#8-1-1} It can be used as CI/CD action in GitHub whenver something is pushed
    2. {#8-2} ```mkinit``` is not the best, as it fills up the main namespace, but still it might be useful as all the classes are available without the submodules
9. #9 Fix requirements.txt and dependencies in setup.py
10. #10 Check for project metadata to be saved in pyproject.toml
    1. #10-1 Reformat requirements.txt to use '.' and install all the required dependencies
11. #11 Use bumpversion or any alternative to automatically bump the version of the repository and create a tag
12. #12 Fix all modules using extra dependencies, dependency must be installed when importing the module for everything to work properly
    1. #12-1 For now we were going through the safe route of checking for imports and raising errors at runtime, but it is not safe
        1. #12-1-1 Check pytorch_lightning.utilities.imports for some tricks
13. {#13} Complete storage_typings with Protocols
14. #14 Remove unnecessary files/code

## (In progress)

## [No Fix]

## {Completed}

1. {#13} Completed storage_typings with Protocols for ExperimentRun, Injection, Fault and Monitor
2. {#8}, {#8-1}, {#8-1-1}, {#8-2} Set up ```mkinit``` in GitHub Actions, to be tested
