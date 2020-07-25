import copy

import torch

def setup_fi(model, *, module_name, fi_class, fi_args={}):
    # we copy the model, to avoid changing the original weights
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))

    fi_instance = fi_class(**fi_args)
    
    # we iterate over all the submodules
    for m in new_model.modules():
        # if one of the modules, including the model itself, have the attribute
        # corresponding to the module_name, it means we have found the module
        # to be injected
        if hasattr(m, module_name):
            # we call the substitution function from the fault injection class
            # ideally it should return a Sequential with the original module
            # and the fault injection at the output, but this is not a strict
            # requirement, e.g. if weights are injected it can be a simple module
            new_module = fi_instance.setup_fi(fi_instance, getattr(m, module_name))
            # we set the new module
            setattr(m, module_name, new_module)
            return new_model
    # we raise an error if no layer has that name
    raise Exception('No layer with name "{}" found'.format(module_name))
