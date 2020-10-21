import abc
import collections.abc

import torch
import pytorch_lightning as pl
try:
    import torchsummary
except ImportError:
    torchsummary = None

# CHECK WHETHER WE WANT THIS INTERDEPENDENCY
try:
    from .. import datasets
except ImportError:
    datasets = None

# FIXME: for now we are using self.logger for logging the training and
# testing steps, but in reality self.logger can be a list containing multiple
# loggers and this case must be handled accordingly

# FIXME: add loggers (TensorBoard) and Python logging

# loss and accuracy functions are functions accepting predictions and targets
# in this order

# metaclass usage for abstract class definition
# or inheritance-based abstract class
class ModuleABC(pl.LightningModule, abc.ABC):
    def __init__(self, input_size=None, output_size=None, *args, **kwargs):
        '''Here we save all the useful settings, like a loss and an accuracy
        functions, accepting predictions and targets in this order.'''
        super().__init__(*args, **kwargs)

        # these dimensions are for supporting custom input/output sizes with
        # fixed model type, e.g. LeNet5 on ImageNet
        # NOTE: not implemented yet
        if input_size is None:
            self._input_size = torch.Size([])
        else:
            self._input_size = torch.Size(input_size)
        if output_size is None:
            self._output_size = torch.Size([])
        else:
            self._output_size = torch.Size(output_size)


        self._summary = None

        # this call is done for saving the object properties, stored in self
        self.save_hyperparameters()


    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        '''This defines the forward method for getting predictions from a test
        input. It is an abstract method, so it cannot be instantiated directly
        but it must be subclassed.'''
        pass

    @staticmethod
    # NOTE: we don't strictly need this feature for running the fault injector,
    # so we try importing torchsummary, setting it to None if not available and
    # returning None if the model_summary function is called
    def model_summary(model, input_size=None):
        # if torchsummary is not available we return None
        if torchsummary is None:
            return None
        # get model and input size, if input is None try inputs from dataset sizes
        # return summary
        if input_size is None and datasets is None:
            raise ValueError('please provide a list as input size for running the summary test')
        elif input_size is None:
            # gather dataset sizes
            # we need a static set of dataset sizes, the one we have now is dynamic
            # solution: use class attributes as defaults, and eventually they are updated dynamically
            # NOTE: current solution uses class attributes,
            # but it could also instantiate the class and get the dynamic
            # one with the default arguments
            input_sizes = [x._size for x in datasets.DATASETS.values() if x._size is not None]
        else:
            # cleanest way to check if we have a list
            # if isinstance(input_size, collections.abc.Iterable) and not isinstance(input_size, str):
            #     input_sizes = input_size
            # else:
            input_sizes = [input_size]

        for size in input_sizes:
            try:
                # slow because of cpu
                device = getattr(model, 'device', 'cpu')
                summary = torchsummary.summary(model, size, device=device)
            # RuntimeError occurs if sizes mismatch in a network, so if that
            # occurs it means the network is not compatible with the dataset
            # size we are using, so we skip it
            except RuntimeError:
                continue
            else:
                return summary

    @property
    def input_size(self):
        ### CHECK
        return self._input_size

    @property
    def output_size(self):
        ### CHECK
        return self._output_size

    @property
    def summary(self):
        if self._summary is None:
            self._summary = self.model_summary(self, self._input_size)
        return self._summary

    # FIXME: change this hardcoded version of constants into a
    # namespace/namedtuple for configuration
    def custom_log_dict(self, dictionary):
        config = dict(prog_bar=True, logger=True, on_epoch=True,
                      reduce_fx=torch.mean, sync_dist=True)

        self.log_dict(dictionary, **config)
