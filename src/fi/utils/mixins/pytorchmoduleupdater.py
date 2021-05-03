import copy
import typing


class PyTorchModuleUpdater(object):
    @classmethod
    def update_module_from_module_list(
            cls,
            target_module: 'torch.nn.Module',
            module_list: typing.Dict[
                    str,
                    'torch.nn.Module'],
            in_place: bool = True,
            ):
        # if in place we modify the module with all the list elements
        # otherwise we copy it and return a copy
        if not in_place:
            target_module = copy.deepcopy(target_module)

        for module_name, module in module_list.items():
            # we reach the final module whose attribute must be updated
            # by going through the tree until the last-but-one attribute
            module_to_be_updated = module_name.split('.')[-1]
            parent_module_name = '.'.join(module_name.split('.')[:-1])

            parent_module = cls.get_module(
                    parent_module_name,
                    target_module
            )

            # the last attribute is used to set the module to the injection
            # one
            setattr(parent_module, module_to_be_updated,
                    module)

        return target_module

    # this function is used to get a target module from its name, using as
    # root the module which is passed as argument
    @classmethod
    def get_module(
            cls,
            target_module_name: str,
            module: 'torch.nn.Module',
            ):
        dest_module = module
        for submodule in target_module_name.split('.'):
            dest_module = getattr(dest_module, submodule)
        return dest_module
