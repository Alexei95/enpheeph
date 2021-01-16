import torchsummary


# this model interface is required for providing the correct info to the
# fault injection manager / experiment, such as layers, sizes, etc.
# for PyTorch, use torchsummary, because PyTorchLightning does not have support
# for MACs, memory, etc.
# FIXME: in future it will be an abstract class providing all the different
# interfaces, such as PyTorch-Lightning, PyTorch, TensorFlow, ...
# FIXME: when providing different interfaces, it should also have a standard
# interface for the different summary classes. For now it is not needed.
class ModelInterface(object):
    def __init__(self, model):
        self._model = model
        self._summary = None

    @property
    def summary(self):
        if self._summary is None:
            self._summary = torchsummary.summary(self._model)
        return self._summary
