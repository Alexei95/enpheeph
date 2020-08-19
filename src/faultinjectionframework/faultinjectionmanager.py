import copy


class FaultInjectionManager(object):
    def setup_fi(self, model):
        # NOTE: we could have used named_modules, but then we would have checked
        # the module directly, making it impossible to update it without knowing
        # its parent
        # we iterate over all the submodules
        for m in model.modules():
            # if one of the modules, including the model itself, have the attribute
            # corresponding to the module_name, it means we have found the module
            # to be injected
            if hasattr(m, self._target_module):
                # first we save the module we have found
                self.backup_module(m, self._target_module_name)
                # we call the substitution function from the fault injection class
                # ideally it should return a Sequential with the original module
                # and the fault injection at the output, but this is not a strict
                # requirement, e.g. if weights are injected it can be a simple module
                new_module = self.update_module(m, self._target_module_name)
                # we set the new module
                setattr(m, self._target_module_name, new_module)

    def restore_fi(self, model):
        # we iterate over all the submodules
        for m in model.modules():
            # when we find a match we run the restoring function which returns
            # the old module configuration to be set in the parent module
            if hasattr(m, self._target_module):
                old_module = self.restore_module(m, self._target_module_name)
                setattr(m, self._target_module_name, old_module)

    def backup_module(self, parent_module, target_module_name):
        original_module = getattr(parent_module, target_module_name)
        module_copy = copy.deepcopy(original_module)
        module_copy.load_state_dict(copy.deepcopy(original_module.state_dict()))
        self._backup_modules[(hash(parent_module), target_module_name)] = module_copy

    def restore_module(self, parent_module, target_module_name):
        old_module_copy = copy.deepcopy(self._backup_modules[(hash(parent_module), target_module_name)])
        old_module_copy.load_state_dict(copy.deepcopy(self._backup_modules[(hash(parent_module), target_module_name)]))
        return old_module_copy
