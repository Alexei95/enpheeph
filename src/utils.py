import os
import pathlib
import sys

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))


# this function sets up the seed for PyTorch / numpy
# if cuda is available it also enables cuDNN deterministic flags
def enable_determinism(seed=42):
    # seed the Python hash generator
    os.environ['PYTHONHASHSEED'] = str(seed)

    # seed the standard Python pRNG
    import random
    random.seed(seed)

    # seed numpy if available
    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(seed)

    # seed pytorch and set determinism for cudnn
    # cuda and cudnn have no effect if cuda is not available
    # NOTE: setting cudnn for determinism can reduce a lot the performance
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

