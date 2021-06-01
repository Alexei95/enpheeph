import builtins
import importlib
import typing


class ImportUtils(object):
    @classmethod
    def get_callable_from_import(cls, element: str) -> typing.Callable:
        # we copy the whole name of the element
        module_string = element
        # we define the module to be None, so that we can check before
        # returning the callable
        module = None
        # we repeat the cycle depending on how many elements we have in
        # the full callable path, e.g. src.fi.utils.Action.get_function
        # will repeat the loop for 5 times
        for i in range(len(element.split('.'))):
            # at each iteration we split the string from the end
            # towards the beginning, using the dot '.' and allowing at most
            # 1 split, hence getting the first element with all the names
            # except the last one, which is the one we are interested in
            # the second element is the function name, so we save it
            # as the callable name
            module_string, callable_string = module_string.rsplit(
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
        # we get the final callable using the callable_string and the
        # imported module
        callable_ = getattr(module, callable_string, None)

        # we raise the error if callable_ is not existing
        if callable_ is None:
            raise ValueError(f"{callable_string} not found in {module_string}")

        return callable_

    # we return a callable from a string, going through the local namespace
    # as well as the builtin functions
    @classmethod
    def get_callable_from_namespace(cls, element: str) -> typing.Callable:
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

    # this function returns a callable from a string and a flag to force
    # its import
    @classmethod
    def get_callable(
            cls,
            element: str,
            import_: bool = False
    ) -> typing.Callable:
        # if we are required to import the callable
        if import_:
            # we get the callable importing the required elements
            callable_ = cls.get_callable_from_import(element)
        # if not required, we cycle through the module and getattr to reach the
        # final callable
        else:
            callable_ = cls.get_callable_from_namespace(element)
        return callable_
