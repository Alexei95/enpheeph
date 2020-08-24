import collections

import torch
import pytorch_lightning as pl

from . import moduleabc
from ..datasets import DATASETS

# FIXME: implementation of input and output sizes can be improved
MNIST = DATASETS.get('MNIST', None)
if MNIST is None:
    DEFAULT_LENET5_INPUT_SIZE = torch.Size([1, 28, 28])
    DEFAULT_LENET5_OUTPUT_SIZE = torch.Size([10])
else:
    DEFAULT_LENET5_INPUT_SIZE = MNIST.size
    DEFAULT_LENET5_OUTPUT_SIZE = torch.Size([MNIST.n_classes])

# this size is required to compare the size of the initial layer to match the
# input dataset with the following layers
# it is computed based on the 28x28 MNIST dataset
DEFAULT_LENET5_C1_OUTPUT_SIZE = moduleabc.ModuleABC.compute_output_dimension(input_size=(28, 28),
                                                                             kernel_size=(3, 3),
                                                                             stride=(1, 1),
                                                                             padding=(0, 0))


class LeNet5(moduleabc.ModuleABC):
    def __init__(self, input_size=DEFAULT_LENET5_INPUT_SIZE, output_size=DEFAULT_LENET5_OUTPUT_SIZE, *args, **kwargs):
        kwargs['input_size'] = input_size
        kwargs['output_size'] = output_size

        super().__init__(*args, **kwargs)

        # this implementation is from the PyTorch implementation in the tutorial
        # FIXME: make the first layer customizable to accept also different
        # datasets, depending on the input dimension
        # the number of channels is the first dimension of the input
        in_channels = self._input_size[0]
        # the number of classes is the first dimension of the output
        n_classes = self._output_size[0]

        c1_kernel_size = self.compute_kernel_dimension(input_size=self._input_size[1:],
                                                       output_size=DEFAULT_LENET5_C1_OUTPUT_SIZE,
                                                       stride=(1, 1),
                                                       padding=(0, 0))

        self.convnet = torch.nn.Sequential(collections.OrderedDict([
            ('c1', torch.nn.Conv2d(in_channels, 6, kernel_size=c1_kernel_size)),
            ('relu1', torch.nn.ReLU()),
            ('s2', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', torch.nn.Conv2d(6, 16, kernel_size=(3, 3))),
            ('relu3', torch.nn.ReLU()),
            ('s4', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        ]))

        self.fc = torch.nn.Sequential(collections.OrderedDict([
            ('c5', torch.nn.Linear(16 * 5 * 5, 120)),
            ('relu5', torch.nn.ReLU()),
            ('f6', torch.nn.Linear(120, 84)),
            ('relu6', torch.nn.ReLU()),
            ('f7', torch.nn.Linear(84, n_classes)),
            ('sig7', torch.nn.LogSoftmax(dim=-1))
        ]))

    # the decorator is to automatically move all the inputs and outputs to the
    # correct device, it has no effect if no LightningModule or not to
    # __call__ or forward
    @pl.core.decorators.auto_move_data
    def forward(self, x, *args, **kwargs):
        output = self.convnet(x)
        output = output.view(x.size(0), -1)
        output = self.fc(output)
        return output

MODEL = {LeNet5.__name__: LeNet5}
