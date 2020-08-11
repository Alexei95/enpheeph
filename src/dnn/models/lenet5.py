import collections

import torch
import pytorch_lightning as pl

from . import basemodule

DEFAULT_LENET5_IN_CHANNELS = 1


class LeNet5(basemodule.BaseModule):
    def __init__(self, in_channels=DEFAULT_LENET5_IN_CHANNELS, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # this implementation is from the PyTorch implementation in the tutorial
        # FIXME: make the first layer customizable to accept also different
        # datasets, depending on the input dimension
        self.convnet = torch.nn.Sequential(collections.OrderedDict([
            ('c1', torch.nn.Conv2d(in_channels, 6, kernel_size=(3, 3))),
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
            ('f7', torch.nn.Linear(84, 10)),
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
