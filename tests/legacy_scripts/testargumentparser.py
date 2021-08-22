import pathlib
import sys

CURRENT_DIR = pathlib.Path(__file__).parent.resolve()
SRC_PARENT_DIR = (CURRENT_DIR / '..').resolve()

sys.path.append(str(SRC_PARENT_DIR))

import enpheeph.cli.utils.argumentparser


argparser = enpheeph.cli.utils.argumentparser.ArgumentParser()
# the following works
# print(argparser.parse_args([]))
# print(argparser.parse_args(['--config']))
print(argparser.parse_args(['--config', 'a']))
print(argparser.parse_args(['-c', 'b']))
