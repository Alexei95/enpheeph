import builtins
import importlib
import typing


class ImportUtils(object):
    @classmethod
    def get_object_from_import(cls, element: str) -> typing.Any:
        # we copy the whole name of the element
        module_string = element
        # we define the module to be None, so that we can check before
        # returning the object
        module = None
        # we repeat the cycle depending on how many elements we have in
        # the full object path, e.g. src.fi.utils.Action.get_function
        # will repeat the loop for 5 times
        for i in range(len(element.split('.'))):
            # at each iteration we split the string from the end
            # towards the beginning, using the dot '.' and allowing at most
            # 1 split, hence getting the first element with all the names
            # except the last one, which is the one we are interested in
            # the second element is the function name, so we save it
            # as the object name
            module_string, object_string = module_string.rsplit(
                    '.',
                    maxsplit=1
            )
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
        object_ = getattr(module, object_string, None)

        # we raise the error if object_ is not existing
        if object_ is None:
            raise ValueError(f"{object_string} not found in {module_string}")

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
