import copy

from . import modelwrapperabc


class PyTorchModelWrapper(modelwrapperabc.ModelWrapperABC):
    @staticmethod
    def copy_module(module):
        copy_ = copy.deepcopy(module)
        copy_.load_state_dict(copy.deepcopy(module.state_dict()))
        return copy_

    def get_module(self, name, first_only=False):
        # NOTE: we could have used named_modules, but then we would have checked
        # the module directly, making it impossible to update it without knowing
        # its parent
        # we iterate over all the submodules

        # if first_only is true, we return only the first matching occurrence
        # inside a list
        # if it is false we return a tuple with all the occurrences
        res = []
        for m in self._model.modules():
            # if one of the modules, including the model itself, have the attribute
            # corresponding to the module_name, it means we have found the module
            # to be injected
            if hasattr(m, name):
                # first we save the module we have found
                res.append(getattr(m, name))
                if first_only:
                    break
        return tuple(res)

    def set_module(self, name, value, first_only=False):
        index = 0
        for m in self._model.modules():
            if hasattr(m, self._target_module):
                setattr(m, value[index])
                if first_only:
                    break
                index += 1


MODEL_WRAPPER = {PyTorchModelWrapper.__name__: PyTorchModelWrapper}
