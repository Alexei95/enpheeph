import builtins
import importlib
import types
import typing


class ImportUtils(object):
    @classmethod
    def get_object_from_import(cls, element: str) -> typing.Any:
        # we copy the whole name of the element
        module_string = element
        # we define the module to be None, so that we can check before
        # returning the object
        # also object_string is empty at the beginning
        module = None
        object_string = ''
        # we iterate from the last sub-object in the string by split
        for el in reversed(element.split('.')):
            # we drop the last element in the whole element string
            # and at each iteration we drop one more
            # to try and import the current module_string
            module_string = module_string.rsplit(
                    '.',
                    maxsplit=1
            )[0]
            # we update the object_string, uniting the elements from last to
            # first, the opposite direction compared to the search for the
            # module
            # if the object_string is empty, we directly copy the last element
            # this is done at the first iteration
            if object_string:
                object_string = '.'.join([el, object_string])
            else:
                object_string = el

            # we try importing the module, if we get an ImportError we
            # continue on the next iteration, otherwise we can stop
            try:
                module = importlib.import_module(module_string)
            except ImportError:
                continue
            else:
                break
        if module is None:
            raise ValueError(f"{element} is not importable")
        # we get the final object using the object_string and the
        # imported module
        object_ = cls.get_object_from_module(object_string, module)

        return object_

    # we return an object from a string, going through the local namespace
    # as well as the builtin functions
    @classmethod
    def get_object_from_namespace(cls, element: str) -> typing.Any:
        # we get the module from the vars, since it is a dict
        # if not present in vars, we check in __builtins__, then we let it
        # raise its error
        # the string is taken from the first element before the dot
        first_string = element.split('.')[0]
        if first_string in vars():
            value = vars()[first_string]
        elif first_string in vars(builtins):
            value = getattr(builtins, first_string)
        else:
            raise ValueError(
                    f'{element} is not found in the current '
                    'namespace or the builtins'
            )
        # then we cycle over the remaining strings from the split
        for element in element.split('.')[1:]:
            # each time we go deeper in the hierarchy by using getattr
            value = getattr(value, element, None)
            if value is None:
                raise ValueError(
                        f"{element} does not exist in the current "
                        "namespace or bultins"
                )

        return value

    # this function returns an object from a string and a flag to force
    # its import
    @classmethod
    def get_object(
            cls,
            element: str,
            import_: bool = False
    ) -> typing.Any:
        # if we are required to import the object
        if import_:
            # we get the object importing the required elements
            object_ = cls.get_object_from_import(element)
        # if not required, we cycle through the module and getattr to reach the
        # final object
        else:
            object_ = cls.get_object_from_namespace(element)
        return object_

    # to return an object from a string and a module
    @classmethod
    def get_object_from_module(
            cls,
            object_string: str,
            module: types.ModuleType
    ) -> typing.Any:
        # we iterate over the children from the module to the final object
        object_ = module
        for el in object_string.split('.'):
            object_ = getattr(object_, el, None)
            # we raise the error if object_ is not existing
            if object_ is None:
                raise ValueError(
                        f"{object_string} not found in {module_string}"
                )
        return object_
