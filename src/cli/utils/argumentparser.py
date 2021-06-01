import argparse
import pathlib


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_argument(
                'config',
                nargs='+',
                action='extend',
                type=pathlib.Path
        )
