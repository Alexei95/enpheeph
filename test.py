import collections
import copy
import functools
import itertools
import logging
import math
import operator
import pathlib
import struct
import sys

import numpy
import pandas
import torch
import torch.nn
import torch.utils.data
import torchvision

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))


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
         # Initialize weights
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
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


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))



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

    def __init__(self):
        super().__init__()

        # the kernels are supposed to be 5x5 with one skipping connection but
        # it's difficult to implement
        self.convnet = torch.nn.Sequential(collections.OrderedDict([
            ('c1', torch.nn.Conv2d(1, 6, kernel_size=(4, 4))),
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

# whole tensor becomes random
class RandomTensorFI(torch.nn.Module):
    def __init__(self, enableFI=True):
        super().__init__()
        self.enableFI = enableFI

    def forward(self, x):
        if self.enableFI:
            r = torch.rand_like(x)
        else:
            r = x
        return r

# a variable percentage of elements become random
class RandomElementFI(torch.nn.Module):
    def __init__(self, enableFI=True, coverage=1.0, n_elements=None):
        super().__init__()
        self.enableFI = enableFI
        self.coverage = coverage
        if (coverage is None and n_elements is None) or (coverage is not None and n_elements is not None):
            raise Exception('Only one of coverage and n_elements must be different from None')
        self.n_elements = n_elements

    def forward(self, x):
        if self.enableFI:
            # total = functools.reduce(operator.mul, x.size())
            # covered = math.ceil(self.percentage * total)
            # index = [tuple(torch.randint(low=0, high=index_dim, size=(1, )).item() for index_dim in x.size()) for _ in range(covered)]
            # r = torch.clone(x).detach()  # add requires_grad_(True) for grad
            # for i in index:
            #     r[i] = torch.randn(size=(1, ))
            r = torch.clone(x).detach().flatten()  # add requires_grad_(True) for grad
            perm = torch.randperm(r.numel())  # we could use cuda but loop is in Python
            if self.coverage is not None and self.n_elements is None:
                covered = math.ceil(self.coverage * r.numel())
            elif self.coverage is None and self.n_elements is not None:
                covered = self.n_elements
            covered = min(max(0, covered), r.numel())
            for i in range(covered):
                r[perm[i]] = torch.randn(size=(1, ), device=x.device)
            r = r.reshape(x.size())
        else:
            r = x
        return r

# a variable percentage of elements has a fixed amount of bits flipped
class RandomBitFlipFI(torch.nn.Module):
    def __init__(self, enableFI=True, coverage=1.0, n_elements=None, n_bit_flips=1):
        super().__init__()
        self.enableFI = enableFI
        if (coverage is None and n_elements is None) or (coverage is not None and n_elements is not None):
            raise Exception('Only one of coverage and n_elements must be different from None')
        self.coverage = coverage
        self.n_elements = n_elements
        self.n_bit_flips = n_bit_flips  # tailored for fp32, can be changed

    def forward(self, x):
        if self.enableFI:
            # total = functools.reduce(operator.mul, x.size())
            # covered = math.ceil(self.percentage * total)
            # index = [tuple(torch.randint(low=0, high=index_dim, size=(1, )).item() for index_dim in x.size()) for _ in range(covered)]
            # r = torch.clone(x).detach()  # add requires_grad_(True) for grad
            # for i in index:
            #     r[i] = torch.randn(size=(1, ))
            r = torch.clone(x).detach().flatten()  # add requires_grad_(True) for grad
            perm = torch.randperm(r.numel())  # we could use cuda but loop is in Python
            if self.coverage is not None and self.n_elements is None:
                covered = math.ceil(self.coverage * r.numel())
            elif self.coverage is None and self.n_elements is not None:
                covered = self.n_elements
            covered = min(max(0, covered), r.numel())
            for i in range(covered):
                r[perm[i]] = bit_flip(r[perm[i]], self.n_bit_flips)
            r = r.reshape(x.size())
        else:
            r = x
        return r


# gets the binary value from a PyTorch element
def pytorch_element_to_binary(value):
    # required because shapes (1, ) and () are considered different and we need ()
    if value.size() != tuple():
        value = value[0]
    # uint to avoid double sign repetition
    data_mapping = {numpy.dtype('float16'): numpy.uint16,
                    numpy.dtype('float32'): numpy.uint32,
                    numpy.dtype('float64'): numpy.uint64}
    width_mapping = {numpy.dtype('float16'): '16',
                     numpy.dtype('float32'): '32',
                     numpy.dtype('float64'): '64'}
    # we get the numpy value, keeping the same datatype
    numpy_value = value.cpu().numpy()
    dtype = numpy_value.dtype
    # we view the number with a different datatype (int) so we can extract the bits
    str_bin_value = '{{:0{}b}}'.format(width_mapping[dtype]).format(numpy_value.view(data_mapping[dtype]))

    return str_bin_value


# original_value is used only for device and datatype conversion
def binary_to_pytorch_element(binary, original_value):
    # required because shapes (1, ) and () are considered different and we need ()
    if original_value.size() != tuple():
        original_value = original_value[0]

    # uint to avoid double sign repetition
    data_mapping = {numpy.dtype('float16'): numpy.uint16,
                    numpy.dtype('float32'): numpy.uint32,
                    numpy.dtype('float64'): numpy.uint64}
    width_mapping = {numpy.dtype('float16'): '16',
                     numpy.dtype('float32'): '32',
                     numpy.dtype('float64'): '64'}

    dtype = original_value.cpu().numpy().dtype

    # we convert the bits to numpy integer through Python int for base 2 conversion
    # then we view it back in the original type and convert it to PyTorch
    # square brackets are for creating a numpy.ndarray for PyTorch
    new_numpy_value = data_mapping[dtype]([int(binary, base=2)]).view(dtype)
    # we use [0] to return a single element
    return torch.from_numpy(new_numpy_value).to(original_value)[0]

def bit_flip(value, n_bit_flips):
    list_str_bin_value = list(pytorch_element_to_binary(value))

    perm = torch.randperm(len(list_str_bin_value))

    for i in range(n_bit_flips):
        # also using ^ 1 to invert
        if list_str_bin_value[perm[i]] == '0':
            list_str_bin_value[perm[i]] = '1'
        elif list_str_bin_value[perm[i]] == '1':
            list_str_bin_value[perm[i]] = '0'

    return binary_to_pytorch_element(''.join(list_str_bin_value), value)


def setup_fi(model, module_name, fi_class, fi_args={}):
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))

    for m in new_model.modules():
        if hasattr(m, module_name):
            new_seq = torch.nn.Sequential(collections.OrderedDict([
                        ('original', getattr(m, module_name)),
                        ('fi', fi_class(**fi_args))]))
            setattr(m, module_name, new_seq)
            return new_model

    raise Exception('No layer with name "{}" found'.format(module_name))






if __name__ == '__main__':

    SEED = 1000
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    cuda = True and torch.cuda.is_available()
    train_batch_size = 128
    test_batch_size = 128

    # fault injection configuration
    fi_class = RandomBitFlipFI
    fi_args = {'coverage': 0.001, 'n_bit_flips': 10}

    # LeNet5
    model_save_file = PROJECT_DIR / 'trained_models' / 'lenet5.pkl'
    model = LeNet5()
    fi_layer = "relu5"
    epochs = 15

    # VGG11 - CIFAR10
    # model_save_file = PROJECT_DIR / 'trained_models' / 'vgg11.pkl'
    # model = vgg11()
    # fi_layer = "pool12"
    # epochs = 30

    model_save_file.parent.mkdir(exist_ok=True, parents=True)

    ############ MNIST
    data_transform_train = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    data_transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
    logging.info('===> Loading MNIST training datasets')
    # MNIST dataset
    training_set = torchvision.datasets.MNIST(
        str(PROJECT_DIR / 'datasets' / 'mnist'), train=True, download=True, transform=data_transform_train)
    # Input pipeline
    training_data_loader = torch.utils.data.DataLoader(
        training_set, batch_size=train_batch_size, shuffle=True, num_workers=1)

    logging.info('===> Loading MNIST testing datasets')
    testing_set = torchvision.datasets.MNIST(
        str(PROJECT_DIR / 'datasets' / 'mnist'), train=False, download=True, transform=data_transform_test)
    testing_data_loader = torch.utils.data.DataLoader(
        testing_set, batch_size=test_batch_size, shuffle=False, num_workers=1)


    ############ CIFAR10
    # data_transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding=4),
    #                                         torchvision.transforms.RandomHorizontalFlip(),
    #                                         torchvision.transforms.ToTensor(),
    #                                         torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    # data_transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                     torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    # logging.info('===> Loading CIFAR10 training datasets')
    # # CIFAR10 dataset
    # training_set = torchvision.datasets.CIFAR10(
    #     str(PROJECT_DIR / 'datasets' / 'cifar10'), train=True, download=True, transform=data_transform_train)
    # # Input pipeline
    # training_data_loader = torch.utils.data.DataLoader(
    #     training_set, batch_size=train_batch_size, shuffle=True, num_workers=1)

    # logging.info('===> Loading CIFAR10 testing datasets')
    # testing_set = torchvision.datasets.CIFAR10(
    #     str(PROJECT_DIR / 'datasets' / 'cifar10'), train=False, download=True, transform=data_transform_test)
    # testing_data_loader = torch.utils.data.DataLoader(
    #     testing_set, batch_size=test_batch_size, shuffle=False, num_workers=1)

    loss = torch.nn.CrossEntropyLoss()
    if cuda:
        model = model.cuda()
    if model_save_file.exists():
        # to load on the correct device
        if cuda:
            model.load_state_dict(torch.load(str(model_save_file), map_location=torch.device('cuda')))
            model = model.cuda()
        else:
            model.load_state_dict(torch.load(str(model_save_file), map_location=torch.device('cpu')))
        model.eval()
        torch.no_grad()
        logging.info('Loading pre-trained model')
    else:
        # training if the model does not exist
        model.train()

        optim = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, epochs + 1):
            logging.info('Epoch: {}'.format(epoch))
            for b, (data, target) in enumerate(training_data_loader, start=1):
                if cuda:
                    data = data.cuda()
                    target = target.cuda()
                optim.zero_grad()
                predictions = model(data)
                computed_loss = loss(predictions, target)
                computed_loss.backward()
                optim.step()

        optim.zero_grad()

        torch.save(model.state_dict(), str(model_save_file))

    # testing
    model.eval()
    avg_loss = 0
    accuracy = 0
    elements = 0
    with torch.no_grad():
        for b, (data, target) in enumerate(testing_data_loader, start=1):
            if cuda:
                data = data.cuda()
                target = target.cuda()
            predictions = model(data)
            avg_loss += loss(predictions, target)
            # print(target)
            # print(predictions)
            # print(torch.argmax(predictions, dim=-1))
            # print(target == torch.argmax(predictions, dim=-1))
            # sys.exit()
            accuracy += torch.sum(target == torch.argmax(predictions, dim=-1))
            elements += target.size()[-1]

    logging.info('Testing loss: {}'.format(avg_loss.item() / len(testing_data_loader)))
    logging.info('Testing accuracy: {}'.format(accuracy.item() / elements))

    # fault-injection
    # new_model = setup_fi(model, fi_layer, fi_class=fi_class, fi_args=fi_args)
    # if cuda:
    #     new_model = new_model.cuda()

    # avg_loss = 0
    # accuracy = 0
    # elements = 0
    # new_model.eval()
    # with torch.no_grad():
    #     for b, (data, target) in enumerate(testing_data_loader, start=1):
    #         if cuda:
    #             data = data.cuda()
    #             target = target.cuda()
    #         predictions = new_model(data)
    #         n_elements = target.size()[-1]
    #         avg_loss += loss(predictions, target) * n_elements
    #         accuracy += torch.sum(target == torch.argmax(predictions, dim=-1))
    #         elements += n_elements

    # logging.info('Testing loss fi: {}'.format(avg_loss.item() / elements))
    # logging.info('Testing accuracy fi: {}'.format(accuracy.item() / elements))

    # dse fault injection
    dataframe = pandas.DataFrame(columns=['layer', 'coverage', 'n_bit_flips', 'accuracy', 'loss'])
    # LeNet5
    layers = ['c1', 'relu1', 's2', 'c3', 'relu3', 's4', 'c5', 'relu5', 'f6', 'relu6', 'f7', 'sig7']
    # VGG11
    # layers = ['conv0', 'act0',
    #           'pool1',
    #           'conv2', 'act2',
    #           'pool3',
    #           'conv4', 'act4',
    #           'conv5', 'act5',
    #           'pool6',
    #           'conv7', 'act7',
    #           'conv8', 'act8',
    #           'pool9',
    #           'conv10', 'act10',
    #           'conv11', 'act11',
    #           'pool12',
    #           'classifier_dropout0', 'classifier_fc0', 'classifier_act0',
    #           'classifier_dropout1', 'classifier_fc1', 'classifier_act1',
    #           'classifier_fc2']
    # coverage
    # min_coverage = 0
    # n_steps = 10
    # stop_coverage = 0.01
    # coverages = numpy.linspace(min_coverage, stop_coverage, n_steps)
    # number of injections
    min_n_elems = 0
    step = 1
    max_n_elems = 15
    coverages = list(range(min_n_elems, max_n_elems + step, step))
    list_bit_flips = list(range(0, 33))
    # layers = ['relu5']
    # coverages = [0.1]
    # list_bit_flips = [1, 2]

    total = len(layers) * len(coverages) * len(list_bit_flips)

    for i, (l, c, b) in enumerate(itertools.product(layers, coverages, list_bit_flips)):

        logging.info('Configuration {}'.format(i + 1))
        logging.info('Layer: {} Coverage: {} #bitflips: {}'.format(l, c, b))

        dataframe.loc[i, 'layer'] = l
        dataframe.loc[i, 'coverage'] = c
        dataframe.loc[i, 'n_bit_flips'] = b

        new_model = setup_fi(model, l, fi_class=fi_class, fi_args={'coverage': None, 'n_bit_flips': b, 'n_elements': c})
        if cuda:
            new_model = new_model.cuda()

        avg_loss = 0
        accuracy = 0
        elements = 0
        new_model.eval()
        with torch.no_grad():
            for b, (data, target) in enumerate(testing_data_loader, start=1):
                if cuda:
                    data = data.cuda()
                    target = target.cuda()
                predictions = new_model(data)
                n_elements = target.size()[-1]
                avg_loss += loss(predictions, target) * n_elements
                accuracy += torch.sum(target == torch.argmax(predictions, dim=-1))
                elements += n_elements

        logging.info('Testing loss fi: {}'.format(avg_loss.item() / elements))
        logging.info('Testing accuracy fi: {}'.format(accuracy.item() / elements))

        dataframe.loc[i, 'loss'] = avg_loss.item() / elements
        dataframe.loc[i, 'accuracy'] = accuracy.item() / elements

        dataframe.to_csv('test_n_elements_LeNet5_v2.csv')
