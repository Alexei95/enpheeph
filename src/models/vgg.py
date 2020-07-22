import collections
import functools

import torch

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

# default input channels, 3 for RGB images (CIFAR10, ImageNet)
DEFAULT_VGG_IN_CHANNELS = 3

class VGG(torch.nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = torch.nn.Sequential(collections.OrderedDict([
            ('classifier_dropout0', torch.nn.Dropout()),
            ('classifier_fc0', torch.nn.Linear(512, 512)),
            ('classifier_act0', torch.nn.ReLU(True)),
            ('classifier_dropout1', torch.nn.Dropout()),
            ('classifier_fc1', torch.nn.Linear(512, 512)),
            ('classifier_act1', torch.nn.ReLU(True)),
            ('classifier_fc2', torch.nn.Linear(512, 10)),
        ]))
        
        # FIXME: check out weight initialization, here is gaussian for weights
        # and zeros for bias, find a way of generalizing
        # Initialize weights
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #         m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# cfg is a list of layer dimensions for convolution, compare it with the 
# variable named DEFAULT_CONFIGS
# batch_norm is for batch normalization
# in_channels represents the number of input channels
def make_layers_vgg(cfg, batch_norm=False, in_channels=DEFAULT_VGG_IN_CHANNELS):
    layers = []
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

def make_vgg(*args, **kwargs):
    return VGG(make_layers_vgg(*args, **kwargs))


vgg11 = functools.partial(make_vgg, cfg=DEFAULT_VGG_CONFIGS['11'])
vgg13 = functools.partial(make_vgg, cfg=DEFAULT_VGG_CONFIGS['13'])
vgg16 = functools.partial(make_vgg, cfg=DEFAULT_VGG_CONFIGS['16'])
vgg19 = functools.partial(make_vgg, cfg=DEFAULT_VGG_CONFIGS['19'])
