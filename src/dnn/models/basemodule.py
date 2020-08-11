import abc

import torch
import pytorch_lightning as pl

# FIXME: for now we are using self.logger for logging the training and
# testing steps, but in reality self.logger can be a list containing multiple
# loggers and this case must be handled accordingly

# FIXME: add loggers (TensorBoard) and Python logging

# loss and accuracy functions are functions accepting predictions and targets
# in this order

# metaclass usage for abstract class definition
# or inheritance-based abstract class
class BaseModule(pl.LightningModule, abc.ABC):
    def __init__(self, loss, accuracy_fnc, optimizer_class, optimizer_args, input_dims=None, output_dims=None, *args, **kwargs):
        '''Here we save all the useful settings, like a loss and an accuracy
        functions, accepting predictions and targets in this order.'''
        super().__init__(*args, **kwargs)

        self._loss = loss
        self._accuracy_fnc = accuracy_fnc
        self._optimizer_class = optimizer_class
        self._optimizer_args = optimizer_args

        self._input_dims = input_dims
        self._output_dims = output_dims

        # this call is done for saving the object properties, stored in self
        self.save_hyperparameters()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        '''This defines the forward method for getting predictions from a test
        input. It is an abstract method, so it cannot be instantiated directly
        but it must be subclassed.'''
        return None

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
