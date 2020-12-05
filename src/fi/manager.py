from .faultmodels import faultmodelabc
from .modelinterfaces import modelinterfaceabc


class FaultInjectionManager(object):
    def __init__(self, fault_model: faultmodelabc.FaultModelABC, model_interface: modelinterfaceabc.ModelInterfaceABC):
        self._fault_model = fault_model
        self._model_interface = model_interface

        self._faults = []

    # this generates different faults to be propagated through the fault_model
    # it is a simple wrapper for the generate_fi_campaign in the fault_model
    # the results which are FaultInjections are passed to the model_interface
    # which will generate the corresponding experiments based on the type of
    # model we have (PyTorch, PyTorch-Lightning, TensorFlow, ...)
    # these are tentative inputs, may change later
    # the return value is a list of experiments
    # each experiment contains the original model with the new fault injection
    # module, and therefore can be run to
    def generate_fi_campaign(self, n_executions, n_coverage):
        pass

    # to allow multiple processes using a single thread
    # here the logic for concurrency should be present
    def run_experiment(self):
        pass
