import abc

import torch
import pytorch_lightning as pl
try:
    import torchsummary
except ImportError:
    torchsummary = None

# CHECK WHETHER WE WANT THIS INTERDEPENDENCY
from .. import datasets

# FIXME: for now we are using self.logger for logging the training and
# testing steps, but in reality self.logger can be a list containing multiple
# loggers and this case must be handled accordingly

# FIXME: add loggers (TensorBoard) and Python logging

# loss and accuracy functions are functions accepting predictions and targets
# in this order

# metaclass usage for abstract class definition
# or inheritance-based abstract class
class ModuleABC(pl.LightningModule, abc.ABC):
    def __init__(self, loss, accuracy_fnc, optimizer_class, optimizer_args, input_size=None, output_size=None, *args, **kwargs):
        '''Here we save all the useful settings, like a loss and an accuracy
        functions, accepting predictions and targets in this order.'''
        super().__init__(*args, **kwargs)

        self._loss = loss
        self._accuracy_fnc = accuracy_fnc
        self._optimizer_class = optimizer_class
        self._optimizer_args = optimizer_args

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
    def compute_kernel_dimension(input_size, output_size, stride=(1, 1),
                                 padding=(0, 0)):
        # output = (input - filter + 2 * padding) / stride + 1
        kernel = []
        for i, o, p, s in zip(input_size, output_size, padding, stride):
            kernel.append(i + 2 * p - (o - 1) * s)
        return tuple(kernel)

    @staticmethod
    def compute_output_dimension(input_size, kernel_size, stride=(1, 1),
                            padding=(0, 0)):
        # output = (input - filter + 2 * padding) / stride + 1
        output = []
        for i, k, p, s in zip(input_size, kernel_size, padding, stride):
            output.append((i - k + 2 * p) / s + 1)
        return tuple(output)

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
        if input_size is None:
            # gather dataset sizes
            # we need a static set of dataset sizes, the one we have now is dynamic
            # solution: use class attributes as defaults, and eventually they are updated dynamically
            # NOTE: current solution uses class attributes,
            # but it could also instantiate the class and get the dynamic
            # one with the default arguments
            input_sizes = [x._size for x in datasets.DATASETS.values() if x._size is not None]
        else:
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

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        predictions = self.forward(x)
        loss = self._loss(predictions, y)
        accuracy = self._accuracy_fnc(predictions, y)

        result = pl.TrainResult(minimize=loss, checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({'train_loss': loss, 'train_accuracy': accuracy},
                        prog_bar=True, logger=True, on_epoch=True,
                        reduce_fx=torch.mean)

        return result

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        predictions = self.forward(x)
        loss = self._loss(predictions, y)
        accuracy = self._accuracy_fnc(predictions, y)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({'test_loss': loss, 'test_accuracy': accuracy},
                        prog_bar=True, logger=True, on_epoch=True,
                        reduce_fx=torch.mean)

        return result

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predictions = self.forward(x)
        loss = self._loss(predictions, y)
        accuracy = self._accuracy_fnc(predictions, y)

        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log_dict({'validation_loss': loss, 'validation_accuracy': accuracy},
                        prog_bar=True, logger=True, on_epoch=True,
                        reduce_fx=torch.mean)

        return result

    def configure_optimizers(self):
        optimizer = self._optimizer_class(self.parameters(), **self._optimizer_args)
        # different optimizers can be passed as a tuple
        return optimizer
