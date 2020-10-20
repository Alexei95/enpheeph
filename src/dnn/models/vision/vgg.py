import collections
import functools

import torch
import pytorch_lightning as pl
import pytorch_lightning.core.decorators as pl_decorators

from .common import *
from . import visionmoduleabc
# to avoid interdependency
try:
    from ...datasets import DATASETS
except ImportError:
    DATASETS = tuple()

# standard VGG configs, 11, 13, 16, 19, number is the output channel size
# for conv, M is max pooling
DEFAULT_VGG_CONFIGS = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
           'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
           512, 512, 'M', 512, 512, 512, 512, 'M'],
}
DEFAULT_VGG_INIT_WEIGHTS = True

IMAGENET = DATASETS.get('ImageNet', None)
if IMAGENET is None:
    DEFAULT_VGG_INPUT_SIZE = torch.Size([3, 227, 227])
    DEFAULT_VGG_OUTPUT_SIZE = torch.Size([1000])
else:
    DEFAULT_VGG_INPUT_SIZE = IMAGENET.size()
    DEFAULT_VGG_OUTPUT_SIZE = torch.Size([IMAGENET.n_classes()])

# we assume the default input size for VGG to be ImageNet 227x227

# this size is required to compare the size of the initial layer to match the
# input dataset with the following layers
# it is computed based on the 227x227 ImageNet dataset
# it is required if we want to use bigger images than what the network
# was designed for
DEFAULT_VGG_C0_INPUT_SIZE = DEFAULT_VGG_INPUT_SIZE[-2:]


class VGG(visionmoduleabc.VisionModuleABC):
    '''
    VGG model
    '''
    # FIXME: missing input and output sizes implementation
    def __init__(self, features, output_size=DEFAULT_VGG_OUTPUT_SIZE, init_weights=DEFAULT_VGG_INIT_WEIGHTS, *args, **kwargs):
        kwargs['output_size'] = output_size
        kwargs['features'] = features
        kwargs['init_weights'] = init_weights

        super().__init__(*args, **kwargs)

        self.features = features
        self.classifier = torch.nn.Sequential(collections.OrderedDict([
            ('classifier_dropout0', torch.nn.Dropout()),
            ('classifier_fc0', torch.nn.Linear(512, 512)),
            ('classifier_act0', torch.nn.ReLU(True)),
            ('classifier_dropout1', torch.nn.Dropout()),
            ('classifier_fc1', torch.nn.Linear(512, 512)),
            ('classifier_act1', torch.nn.ReLU(True)),
            ('classifier_fc2', torch.nn.Linear(512, *self.output_size)),
        ]))

        if init_weights:
            self._initialize_weights()

        # FIXME: check out weight initialization, here is gaussian for weights
        # and zeros for bias, find a way of generalizing
        # Initialize weights
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()

    # the decorator is to automatically move all the inputs and outputs to the
    # correct device, it has no effect if no LightningModule or not to
    # __call__ or forward, and it is generally automatically applied
    @pl_decorators.auto_move_data
    def forward(self, x, *args, **kwargs):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


# cfg is a list of layer dimensions for convolution, compare it with the
# variable named DEFAULT_VGG_CONFIGS
# batch_norm is for batch normalization
# in_channels represents the number of input channels
def make_layers_vgg(cfg, batch_norm=False, input_size=DEFAULT_VGG_INPUT_SIZE, convert_input=DEFAULT_CONVERT_INPUT):
    layers = []
    in_channels = input_size[0]
    # this convolutional layer resizes the input if larger than the expected
    # one, to avoid analyzing only a patch of it
    # TODO: if smaller, it should be done with an appropriate up-scaling
    #       look up deconvolution in PyTorch
    if input_size[1:] != DEFAULT_VGG_C0_INPUT_SIZE and convert_input:
        c0_kernel_size = visionmoduleabc.VisionModuleABC.compute_kernel_dimension(
            input_size=input_size[1:],
            output_size=DEFAULT_VGG_C0_INPUT_SIZE,
            stride=(1, 1),
            padding=(1, 1))
        conv2d = torch.nn.Conv2d(in_channels, in_channels, kernel_size=c0_kernel_size, padding=1)
        layers.append(conv2d)
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [('pool{}'.format(i), torch.nn.MaxPool2d(kernel_size=2, stride=2))]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [('conv{}'.format(i), conv2d),
                           ('bn{}'.format(i), torch.nn.BatchNorm2d(v)),
                           ('act{}'.format(i), torch.nn.ReLU(inplace=True))]
            else:
                layers += [('conv{}'.format(i), conv2d),
                           ('act{}'.format(i), torch.nn.ReLU(inplace=True))]
            in_channels = v
    return torch.nn.Sequential(collections.OrderedDict(layers))


# wrapper to return the correct instance of the network class
def make_vgg(cfg, batch_norm=False, input_size=DEFAULT_VGG_INPUT_SIZE, output_size=DEFAULT_VGG_OUTPUT_SIZE, convert_input=DEFAULT_CONVERT_INPUT, *args, **kwargs):
    return VGG(make_layers_vgg(cfg=cfg, batch_norm=batch_norm, input_size=input_size, convert_input=convert_input), output_size=output_size, *args, **kwargs)


VGG11 = functools.partial(make_vgg, cfg=DEFAULT_VGG_CONFIGS['11'])
# __name__ required for using the same format as other models
VGG11.__name__ = 'VGG11'
VGG13 = functools.partial(make_vgg, cfg=DEFAULT_VGG_CONFIGS['13'])
VGG13.__name__ = 'VGG13'
# ImageNet-compatible, size: 3, 224, 224
VGG16 = functools.partial(make_vgg, cfg=DEFAULT_VGG_CONFIGS['16'])
VGG16.__name__ = 'VGG16'
VGG19 = functools.partial(make_vgg, cfg=DEFAULT_VGG_CONFIGS['19'])
VGG19.__name__ = 'VGG19'

MODEL = {VGG11.__name__: VGG11,
         VGG13.__name__: VGG13,
         VGG16.__name__: VGG16,
         VGG19.__name__: VGG19,
         }
