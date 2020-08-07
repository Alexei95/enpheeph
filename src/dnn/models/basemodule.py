import abc

import torch
import pytorch_lightning as pl

# FIXME: for now we are using self.logger for logging the training and
# testing steps, but in reality self.logger can be a list containing multiple
# loggers and this case must be handled accordingly

# FIXME: add loggers (TensorBoard) and Python logging

class BaseModule(pl.LightningModule):

    def __init__(self, loss, optimizer_class, optimizer_args, *args, **kwargs):
        '''Here we save all the useful settings, like a loss which must be a
        function, accepting predictions and targets in this order.'''
        super().__init__(*args, **kwargs)

        self._loss = loss
        self._optimizer_class = optimizer_class
        self._optimizer_args = optimizer_args

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

        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        predictions = self.forward(x)
        loss = self._loss(predictions, y)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        predictions = self.forward(x)
        loss = self._loss(predictions, y)
        return loss

    def configure_optimizers(self):
        optimizer = self._optimizer_class(self.parameters(), **self._optimizer_args)
        # different optimizers can be passed as a tuple
        return optimizer
