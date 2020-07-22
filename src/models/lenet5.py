import collections

import torch

DEFAULT_LENET5_IN_CHANNELS = 1

class LeNet5(torch.nn.Module):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """

    def __init__(self, in_channels=DEFAULT_LENET5_IN_CHANNELS):
        super().__init__()

        # the kernels are supposed to be 5x5 with one skipping connection but
        # it's difficult to implement
        self.convnet = torch.nn.Sequential(collections.OrderedDict([
            ('c1', torch.nn.Conv2d(in_channels, 6, kernel_size=(4, 4))),
            ('relu1', torch.nn.ReLU()),
            ('s2', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', torch.nn.Conv2d(6, 16, kernel_size=(4, 4))),
            ('relu3', torch.nn.ReLU()),
            ('s4', torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', torch.nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', torch.nn.ReLU())
        ]))

        self.fc = torch.nn.Sequential(collections.OrderedDict([
            ('f6', torch.nn.Linear(120, 84)),
            ('relu6', torch.nn.ReLU()),
            ('f7', torch.nn.Linear(84, 10)),
            ('sig7', torch.nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(x.size(0), -1)
        output = self.fc(output)
        return output
