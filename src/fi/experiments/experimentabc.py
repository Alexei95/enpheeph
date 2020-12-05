import abc


# the experiment interface is instantiated with the current model
# and it acts as a wrapper for the model to insert the fault injection and
# run the model in the proper way
class ExperimentABC(object):
    _implementations = []

    # we need a way of discovering all the possible experiment implementations
    # FIXME: check what to do
    def discover_implementations(self):
        pass

    # add a fault to the current experiment
    @abc.abstractmethod
    def add_fault(self, fault: , *args, **kwargs):
        pass

    # using this method we create a new experiment, adding all the faults
    # passed as well as the model and the dataset
    @abc.abstractmethod
    @classmethod
    def make_experiment(cls, *args, **kwargs):
        pass

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass
