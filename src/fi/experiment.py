
# this class implements the basic functions for running a single experiment
# basically it calls the model with the proper parameters to run the experiment
# it should also gather basic info from the run itself, like the injection
# result
# FIXME: it can be generalized using an abstract class to handle the different
# implementations for PyTorch, TensorFlow, ...
class Experiment(object):
    pass
