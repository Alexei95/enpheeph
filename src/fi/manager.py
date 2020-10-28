
class FaultInjectionManager(object):
    def __init__(self, source, fault_model, model_interface):
        self._source = source
        self._fault_model = fault_model
        self._model_interface = model_interface

        self._faults = []

    def generate_faults(self, n_executions):
        pass

    def generate_experiments(self, experiment_class):
        pass

    # to allow multiple processes using a single thread
    def run_experiment(self, index, new_process):
        pass
