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

import src.datasets.mnist
import src.datasets.utils
import src.fi.injectors.randombitflip
import src.models.lenet5
import src.utils


SEED = 1000
USE_CUDA = True and torch.cuda.is_available()
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

# fault injection configuration
FI_CLASS = src.fi.injectors.randombitflip.RandomBitFlipFI
FI_ARGS = {'coverage': 0.001, 'n_bit_flips': 10}
TARGET_LAYER = 'c1'

TRAINING_EPOCHS = 10
MODEL = src.models.lenet5.LeNet5()
DATASET_PATH = PROJECT_DIR / 'datasets'
TRAIN_DATASET, TEST_DATASET = src.datasets.mnist.mnist(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_batch_size, path=DATASET_PATH)


def main():
    src.utils.enable_determinism(SEED)

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
