import copy
import importlib
import os
import pathlib
import types
import typing


class SubclassGatherer(object):
    MODULE_GLOB = '*.py'
    NOT_MODULE_FILTER = ['__init__.py']

    @classmethod
    def gather_subclasses(
            module: types.ModuleType,
            baseclass: type
    ) -> typing.Dict[str, type]:
        # to gather the possible subclasses, we get all the values
        # from the variables inside the module
        # we use vars instead of dir as dir returns the strings, instead vars
        # maps the names to the object themselves
        possible_subclasses = vars(module).values()

        # we initialize the return dict and start looping over the variables
        subclasses = {}
        for candidate in possible_subclasses:
            # we set the flag
            subclass_flag = False
            # if it raises AttributeError it means it doesn't have an mro, so
            # we skip
            try:
                subclass_flag = baseclass in candidate.mro()
            except AttributeError:
                continue
            # we check the flag, if true we get the class name from qualname
            # and we make it full lower-case
            if subclass_flag:
                candidate_name = candidate.__qualname__.lower()
                subclasses[candidate_name] = candidate

        return subclasses
