# TODOs

1. {#1} Add typing.Protocol implementation for ExperimentRun and Injection, so that they can be used as return type in StoragePluginABC
2. #2 Implement tests for all of the plugins and interfaces
3. {#3} Fix all the CI/CD settings
    1. {#3-1} Improve pre-commit and tool configurations
        1. {#3-1-1} Use pytorch_lightning as example
    2. {#3-2} Setup pygrep-hooks to fix flake/mypy/black ignores in ```__init__.py``` mostly
    3. #3-3 Add markdown linter and other optional linters for Python (pydocstyle, ...) to pre-commit as well as GitHub Actions
4. #4 Update all the code to follow black and flake8
5. #5 Add documentation/examples on how to use the different injections
    1. #5-1 Add docstrings and more comments inside the code
    2. #5-2 Use Sphinx and reST for documents, follow pytorch_lightning for some examples
        1. #5-2-1 Docs can be hosted for free on readthedocs.io
6. #6 Add PEP-561 compliance (py.typed)
7. #7 Check for MANIFEST.in
8. {#8} Write proper ```__init__.py``` to import the classes from the sub-modules
    1. {#8-1} ```mkinit``` does that, run ```mkinit --recursive --black --lazy src/enpheeph -w```
        1. {#8-1-1} It can be used as CI/CD action in GitHub whenver something is pushed
    2. {#8-2} ```mkinit``` is not the best, as it fills up the main namespace, but still it might be useful as all the classes are available without the submodules
9. #9 Fix requirements.txt and dependencies in setup.py
10. #10 Check for project metadata to be saved in pyproject.toml
    1. #10-1 Reformat requirements.txt to use '.' and install all the required dependencies
11. #11 Use bumpversion or any alternative to automatically bump the version of the repository and create a tag
12. {#12} Fix all modules using extra dependencies, dependency must be installed when importing the module for everything to work properly
    1. {#12-1} For now we were going through the safe route of checking for imports and raising errors at runtime, but it is not safe
        1. {#12-1-1} Check pytorch_lightning.utilities.imports for some tricks
13. {#13}|#1| Complete storage_typings with Protocols
14. {#14} Remove unnecessary files/code
15. #15 Use towncrier for changelogs and updates
16. #16 Move all info in setup.py and load them in ```__about__``` using ```importlib.metadata```
17. #17 Wait for pyproject.toml support in setuptools before moving everything from setup.py to pyproject.toml
18. #18 Implement logging throughout the code.

## |Duplicates|

1. #13 duplicates #1

## (In progress)

## [No Fix]

## {Completed}

1. {#13}, {#1} Completed storage_typings with Protocols for ExperimentRun, Injection, Fault and Monitor
2. {#8}, {#8-1}, {#8-1-1}, {#8-2} Set up ```mkinit``` in GitHub Actions
3. {#14} Removed unnecessary files
4. {#12}, {#12-1}, {#12-1-1} Fix imports with requirement comparisons and flags. mypy imports are done with typing.TYPE_CHECKING
5. {#3}, {#3-1}, {#3-1-1}, {#3-2} CI/CD has been improved, many more checks are now run both in pre-commit and in CI. There are still some possible improvements to be made, mentioned in #3-3
