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
9. {#9} Fix requirements.txt and dependencies in setup.py
10. {#10} Check for project metadata to be saved in pyproject.toml
    1. {#10-1} Reformat requirements.txt to use '.' and install all the required dependencies
11. #11 Use bumpversion or any alternative to automatically bump the version of the repository and create a tag
12. {#12} Fix all modules using extra dependencies, dependency must be installed when importing the module for everything to work properly
    1. {#12-1} For now we were going through the safe route of checking for imports and raising errors at runtime, but it is not safe
        1. {#12-1-1} Check pytorch_lightning.utilities.imports for some tricks
13. {#13}|#1| Complete storage_typings with Protocols
14. {#14} Remove unnecessary files/code
15. #15 Use towncrier for changelogs and updates
16. #16 Move all info in setup.py and load them in ```__about__``` using ```importlib.metadata```
17. {#17}|#10| Wait for pyproject.toml support in setuptools before moving everything from setup.py to pyproject.toml
18. #18 Implement logging throughout the code.
    1. #18-1 Just write some ```logger = logging.getLogger(__name__)``` and ```logger.debug(...)``` throughout the code
19. #19 Implement early stopping callback inheriting from PyTorch Lightning but checking at each batch
20. #20 Add jax support for CPU/GPU/TPU operations, as it supports dlpack
    1. #20-1 Add support for automatic switching of the mask support across NumPy/CuPy/JAX for CPU/GPU/TPU support
    2. #20-2 Improve mask plugins, as we can have bit-level plugins and tensor-level plugins interacting with each others, as we can use DLPack as intermediary for both NumPy/CuPy/JAX and PyTorch/TensorFlow/JAX, ...
        1. #20-2-1 However, this is limited to very recent versions, like Numpy 1.23.0 and PyTorch 1.11
21. #21 Add support for bit_index in MonitorLocation, so that we can save only some bits
22. #22 Add support for a registry mapping each set of parameter type / injection to a possible class
    1. #22-1 Follow the registry used in PyTorch Lightning Flash
23. {#23} Bug with mask dimension error for activations, as the mask also covers the batch size, so it should be enough to remove it from the mask dimensions
    1. {#23-1} This can be further improved using an indexing system to select the correct indices by means of a dict instead of tensor_index, batch_index, time_index and so on. In this way the different injections can select the indices that they need at each specific time, by only knowing how to reach the generic batch/tensor/time positioning via enums. Even bit_index might be added later on, but since it is generic enough to cover all the implementations it might not be necessary.
24. #24 Use tox for tests automation.
25. #25 Add caching so that we can skip recomputing the layers up until the fault if the checksum of the input is the same
    1. #25-1 Look at InjectTF2 for solutions on how to do it
26. {#26} Fix timestamps/runtime for injections
27. #27 Fix golden run id referring to the first one if the database is being reused
28. #28 Improve docs in indexing plugin
29. #29 Find better solution for warning suppression in InjectionCallback
30. #30 Make SQLStoragePluginABC into a real ABC and move all the implementations in SQLStoragePluginMixin
31. #31 Fix spiking injection
32. #32 Add memory profiling
33. {#33} Add GPG signature for commits
34. #34 Switch to a custom-federated Gitea instance
35. #35 Find a way of auto-updating the dependencies found in GitHub Actions, e.g. mkinit, which is not mentioned anywhere else
    1. #35-1 Additionally consider the introduction of automation bots like DependaBot and similar
36. #36 Improve bit_index behaviour using BitIndexInfo and MSB/LSB Enums for relative computations.
37. {#37} Fix bug with injection working too much for GPU vs CPU
38. {#38} Fix bug on multiple injections not working in sequence
39. #39 Add regression tests for #37, #38
40. {#40} Implement a CI for updating the copyright year
    1. {#40-1} It can be done using insert-license CI from pre-commit, using the CLI option --remove-header to remove the current license notice, than modifying the license and re-running insert-license without --remove-header
    2. {#40-2} There is a GitHub Action for that, which creates a PR that can be automatically merged ``FantasticFiasco/action-update-license-yearFantasticFiasco/action-update-license-year``
41. {#41} Expand SkipIfError to support a tuple of errors
42. #42 Possible issues with PyTorch 1.10 due to layer parametrization, e.g., functions that are run on weight/other attributes, and modify ``.weight`` to be a property and be recomputed from ``<module>.parametrizations.weight.original`` every time the property is called
    1. #42-1 Parametrization might also be fun for implementing injections in future
43. #43 Add tests for ``add_injections`` and ``remove_injections``
44. #44 Implement ``dimension_masked_index``, to allow masking instead of indexing. Masking works well for selecting patterns which are not easily translatable into broadcastable indices. Additionally, it can be mixed with ``dimension_index`` for different dimensions to allow both indexing and masking. However, they need not to overlap, as this would cause problem.
    1. #44-1 The solution is using both as optional and overlapping them, but at least one must be provided
    2. #44-2 Implement tests
    3. #44-3 Add support for batch mask as well
    4. #44-4 Add support for masks in Monitors
45. #45 Fix execution of Fault for Linear layers in SNNs
    1. #45-1 It should be able to inject in the potential before the threshold operation as well
46. #46 Improve implementation of FPQuantizedOutputPyTorchFault
    1. #46-1 Use a mixin as for DenseSparse

## |Duplicates|

1. #13 duplicates #1
2. #17 duplicates #10

## (In progress)

## [No Fix]

## {Completed}

1. {#13}, {#1} Completed storage_typings with Protocols for ExperimentRun, Injection, Fault and Monitor
2. {#8}, {#8-1}, {#8-1-1}, {#8-2} Set up ```mkinit``` in GitHub Actions
3. {#14} Removed unnecessary files
4. {#12}, {#12-1}, {#12-1-1} Fix imports with requirement comparisons and flags. mypy imports are done with typing.TYPE_CHECKING
5. {#3}, {#3-1}, {#3-1-1}, {#3-2} CI/CD has been improved, many more checks are now run both in pre-commit and in CI. There are still some possible improvements to be made, mentioned in #3-3
6. {#9} Dependencies are now better managed
7. {#23}, {#23-1} Now we can select also batches depending on the corresponding index and the flag, and also the mask is saved accordingly. The corresponding indexing plugin has been implemented and integrated with the common faults, some tests will have to be written to guarantee proper operation.
8. {#26} InjectionCallback computes start and stop times and gives them to the storage plugin
9. {#37} The cause was the mask not being properly instantiated as it was missing the first "batch" dimension. It is now fixed with better indexing and proper copies of the mask on CuPy
10. {#38} SQL was giving errors of multiple objects, the culprits were the use of ExperimentRun.id_ instead of Injection.experiment_run_id as well as the improper checks in integrations with PyTorch Lightning, adding the same injection twice to the list. It is fixed now with using dict keys to remove duplicates.
11. {#33} Unfortunately it can only be committed using ``git commit`` from the terminal
12. {#41} SkipIfError now supports tuples of Exceptions
13. {#10}, {#10-1} Fixed with setuptools >= 61.0, as they support pyproject.toml as setup configuration. setup.py is still required for editable installs in pip.
14. {#40}, {#40-1}, {#40-2} The CI has been implemented but it is not easily testable
