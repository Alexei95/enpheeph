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

# FIXME: improve imports
from src.common import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, PACKAGE_DIR, PROJECT_DIR
import src.dnn.datasets
import src.dnn.datasets.utils
import src.dnn.models
import src.dnn.utils
import src.dnn.test
import src.fi.injectors
import src.fi.samplers
import src.fi.setup
import src.utils


SEED = src.common.DEFAULT_PRNG_SEED
USE_CUDA = True and src.common.USE_CUDA
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

# fault injection configuration
COVERAGE_SAMPLER = src.fi.samplers.SAMPLERS['UniformSampler'](seed=SEED)
ELEMENT_SAMPLER = src.fi.samplers.SAMPLERS['UniformSampler'](seed=SEED)
FI_CLASS = src.fi.injectors.FAULT_INJECTORS['RandomBitFlipFI']
FI_ARGS = {'coverage': 0.001,
           'n_bit_flips': 10,
           'coverage_sampler': COVERAGE_SAMPLER,
           'element_sampler': ELEMENT_SAMPLER}
TARGET_LAYER = 'c1'

TRAINING_EPOCHS = 30
MODEL_CLASS = src.dnn.models.MODELS['LeNet5']
MODEL_ARGS = {}
LOSS = torch.nn.CrossEntropyLoss()
OPTIMIZER_CLASS = torch.optim.Adam
OPTIMIZER_ARGS = {'lr': 0.001}
MODEL_SAVE_FILE = DEFAULT_MODEL_PATH / 'lenet5.pkl'
USE_SAVED_MODEL = True
DATASET_PATH = DEFAULT_DATASET_PATH
DATASET_INIT = src.dnn.datasets.DATASETS['MNIST']


def main():
    # we enable determinism
    src.utils.enable_determinism(SEED)

    # FIXME: improve logging
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    train_dataset, test_dataset = DATASET_INIT(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, path=DATASET_PATH)

    model_init = src.dnn.utils.init_model(MODEL_CLASS, MODEL_ARGS, use_saved_model=USE_SAVED_MODEL,
                                        model_save_path=MODEL_SAVE_FILE, cuda=USE_CUDA,
                                        train_dataset=train_dataset, optimizer_class=OPTIMIZER_CLASS,
                                        optimizer_args=OPTIMIZER_ARGS, loss=LOSS, n_epochs=TRAINING_EPOCHS)

    # FIXME: improve logging
    if model_init['loaded'] and not model_init['trained']:
        logging.info('Model loaded correctly')
    elif not model_init['loaded'] and model_init['trained']:
        logging.info('Training loss: {}'.format(model_init['loss']))
        logging.info('Training accuracy: {}'.format(model_init['accuracy']))
    else:
        logging.error('Error')
        raise Exception()

    # FIXME: saving and loading can be implemented with custom functions
    MODEL_SAVE_FILE.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model_init['model'].state_dict(), str(MODEL_SAVE_FILE))

    golden_testing_res = src.dnn.test.test(model_init['model'], test_dataset, LOSS, cuda=USE_CUDA)

    # FIXME: improve logging
    logging.info('Testing loss: {}'.format(golden_testing_res['loss']))
    logging.info('Testing accuracy: {}'.format(golden_testing_res['accuracy']))

    # fault-injection
    fi_model = src.fi.setup.setup_fi(model_init['model'], module_name=TARGET_LAYER, fi_class=FI_CLASS, fi_args=FI_ARGS)

    fi_testing_res = src.dnn.test.test(fi_model, test_dataset, LOSS, cuda=USE_CUDA)

    # FIXME: improve logging
    logging.info('FI Testing loss: {}'.format(fi_testing_res['loss']))
    logging.info('FI Testing accuracy: {}'.format(fi_testing_res['accuracy']))

if __name__ == '__main__':
    main()
