import pathlib

import numpy.random
import torch.cuda


# CUDA PyTorch settings
CUDA_IS_AVAILABLE = torch.cuda.is_available()
USE_CUDA = True and CUDA_IS_AVAILABLE

# default settings for numpy prng
DEFAULT_PRNG_SEED = 42
DEFAULT_PRNG = numpy.random.PCG64

# default directory settings
# main project dir
PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
# package dir
PACKAGE_DIR = PROJECT_DIR / 'src'
# main datasets directory
DEFAULT_DATASET_PATH = PROJECT_DIR / 'datasets'
# main directory for saving models
DEFAULT_MODEL_PATH = PROJECT_DIR / 'models'