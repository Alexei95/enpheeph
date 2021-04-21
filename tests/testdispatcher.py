import copy
import pprint
import pathlib
import sys

import torch
import torchvision

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SRC_PARENT_DIR = (CURRENT_DIR / '..').resolve()

sys.path.append(str(SRC_PARENT_DIR))

import src.dispatcherabc


class DispatcherTest(src.dispatcherabc.DispatcherABC):
    pass


class DispatcherTest2(src.dispatcherabc.DispatcherABC):
    pass


@DispatcherTest.register_decorator()
def test(pippo):
    print(pippo)


print(DispatcherTest.get_dispatching_dict())
print(DispatcherTest2.get_dispatching_dict())

DispatcherTest.dispatch_call('test', 'test pippo')
