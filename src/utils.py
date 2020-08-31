import copy
import datetime
import enum
import importlib
import os
import pathlib
import sys

# PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
# if str(PROJECT_DIR) not in sys.path:
#     sys.path.append(str(PROJECT_DIR))
from .common import DEFAULT_PRNG_SEED, DEFAULT_TIME_FORMAT


### time handling functions ###

# returns current utctime
def current_utctime():
    return datetime.datetime.now(datetime.timezone.utc)


# returns a string version of the current utc time
def current_utctime_string(template=DEFAULT_TIME_FORMAT):
    return time_string(current_utctime(), template)


# converts localtime to utctime
# it should work in more or less all cases
def localtime_to_utctime(datetime_obj):
    localtime_timestamp = datetime_obj.timestamp()
    return datetime.datetime.fromtimestamp(localtime_timestamp,
                                           tz=datetime.timezone.utc)


# returns the string of a datetime object given the format
def time_string(datetime_obj, template=DEFAULT_TIME_FORMAT):
    return datetime_obj.strftime(template)

### end time handling functions ###

# this function sets up the seed for PyTorch / numpy
# if cuda is available it also enables cuDNN deterministic flags
# this function is similar to pytorch_lightning.seed_everything, but they
# don't set CUDA and cuDNN for determinism
# CUDA and cuDNN determinism can be set from the Trainer class
def enable_determinism(seed=DEFAULT_PRNG_SEED):
    # seed the Python hash generator
    os.environ['PYTHONHASHSEED'] = str(seed)

    # seed the standard Python pRNG
    import random
    random.seed(seed)

    # seed numpy if available
    # NOTE: this is a legacy function, as of 1.17 Numpy uses new generators and
    # each of them has its own seed. However this still works when calling
    # functions from numpy.random instead of the generator
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

# this function gather objects from different files in the same directory
# these objects are squashed together with the update_function, starting from a
# copy of the default_obj
def gather_objects(*, path, filter_, package_name, obj_name, default_obj, update_function, glob='*.py'):
    res = copy.deepcopy(default_obj)
    # we update the result to contain all the name-class associations in the
    # package
    for m in pathlib.Path(path).glob(glob):
        # if the file is __init__.py or a directory we skip
        if m.name in filter_:
            continue
        # we get the full name removing the suffix
        module_name = str(m.with_suffix('').name)
        # we append the package of __init__ for the import
        # FIX: without this package we would be unable to reach the module using
        # relative imports or relative imports inside the module would fail because
        # the module would not know its parent package
        complete_module_name = package_name + '.' + module_name
        # we must also pass the package when importing
        module = importlib.import_module(complete_module_name, package=package_name)
        res = update_function(res, getattr(module, obj_name, default_obj))
        del module
    return res


# this function is used as an external function for joining dict copies
# together, it is even compatible with Python 2
# in Python 3.9+ it could be replaced using |
def update_dicts(dict1, dict2):
    res = copy.deepcopy(dict1)
    res.update(copy.deepcopy(dict2))
    return res


# NOTE: recipe for Ordered Enumerations in Python docs
# https://docs.python.org/3.8/library/enum.html#orderedenum
class OrderedEnum(enum.Enum):

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
