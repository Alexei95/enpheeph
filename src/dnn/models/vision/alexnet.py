import collections

import torch
import pytorch_lightning as pl
import pytorch_lightning.core.decorators as pl_decorators

from . import visionmoduleabc
# to avoid interdependency
try:
    from ...datasets import DATASETS
except ImportError:
    DATASETS = tuple()

# FIXME: implementation of input and output sizes can be improved
ImageNet = DATASETS.get('ImageNet', None)
if ImageNet is None:
    DEFAULT_ALEXNET_INPUT_SIZE = torch.Size([3, 227, 227])
    DEFAULT_ALEXNET_OUTPUT_SIZE = torch.Size([1000])
else:
    DEFAULT_ALEXNET_INPUT_SIZE = ImageNet.size()
    DEFAULT_ALEXNET_OUTPUT_SIZE = torch.Size([ImageNet.n_classes()])

# this size is required to compare the size of the initial layer to match the
# input dataset with the following layers
DEFAULT_ALEXNET_C1_OUTPUT_SIZE = visionmoduleabc.VisionModuleABC.compute_output_dimension(input_size=(227, 227),
                                                                                          kernel_size=(11, 11),
                                                                                          stride=(4, 4),
                                                                                          padding=(2, 2))


class AlexNet(visionmoduleabc.VisionModuleABC):
    def __init__(self, input_size=DEFAULT_ALEXNET_INPUT_SIZE, output_size=DEFAULT_ALEXNET_OUTPUT_SIZE, *args, **kwargs):
        # to add missing arguments to kwargs, in this way we can have different defaults
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
                                                       output_size=DEFAULT_ALEXNET_C1_OUTPUT_SIZE,
                                                       stride=(1, 1),
                                                       padding=(0, 0))

        # input needs to be at least 3x227x227, so that the following dimension
        # is 64x56x56
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=c1_kernel_size, stride=4, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(64, 192, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, n_classes),
        )

    # the decorator is to automatically move all the inputs and outputs to the
    # correct device, it has no effect if no LightningModule or not to
    # __call__ or forward
    @pl_decorators.auto_move_data
    def forward(self, x, *args, **kwargs):
        output = self.features(x)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.classifier(output)
        return output


MODEL = {AlexNet.__name__: AlexNet}
