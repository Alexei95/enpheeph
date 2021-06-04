import copy
import os
import pathlib
import shutil
import sys
import typing

import pytorch_lightning.callbacks


class SaveConfigCallback(pytorch_lightning.callbacks.Callback):
    CONFIG_PATHS = {}
    KWARGS_DICT = {
        "configs": {
            "__custom__": True,
            "__custom_decoder__": "callable",
            "__import__": True,
            "__callable__": "collections.UserList",
            "__args__": [],
            "__kwargs__": {
                "initlist": {
                    "__custom__": True,
                    "__custom_decoder__": "callable",
                    "__import__": True,
                    "__callable__": "collections.UserDict.values",
                    "__args__": [],
                    "__kwargs__": {
                        "self":
                            CONFIG_PATHS,
                    }
                }
            }
        },
        "dest_dir": ".",
        "subdirectory": "configs",
    },
    DICT_CONFIG = {
        "trainer": {
            "__kwargs__": {
                "callbacks": {
                    "__kwargs__": {
                        "initlist": {
                            "__kwargs__": {
                                "self": {
                                    "save_config_callback": {
                                        "__custom__": True,
                                        "__custom_decoder__": "callable",
                                        "__import__": True,
                                        "__callable__": (
                                            "src.dnn.pl.utils.callbacks."
                                            "saveconfigcallback."
                                            "SaveConfigCallback"
                                        ),
                                        "__args__": [],
                                        "__kwargs__": KWARGS_DICT,
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
    }

    def __init__(
            self,
            configs: typing.Sequence[typing.Union[
                    pathlib.Path,
                    str,
                    bytes,
                    os.PathLike
            ]],
            dest_dir: typing.Union[pathlib.Path, str, bytes, os.PathLike],
            subdirectory: str = 'configs',
    ):
        super().__init__()

        self.configs = [pathlib.Path(c) for c in configs]
        self.dest_dir = pathlib.Path(dest_dir)
        self.subdirectory = subdirectory

    def make_config(self):
        self.CONFIG_PATHS = {
                index: str(c.resolve())
                for index, c in enumerate(self.configs)
        }
        self.KWARGS_DICT['dest_dir'] = str(self.dest_dir.resolve())
        self.KWARGS_DICT['subdirectory'] = self.subdirectory

    def on_train_start(
            self,
            trainer,
            pl_module
    ) -> None:
        # we generate the complete destination file
        # and we make sure it exists
        complete_dest_dir = self.dest_dir.resolve() / self.subdirectory
        complete_dest_dir.mkdir(exist_ok=True, parents=True)
        # we cycle through the configurations
        for index, config in enumerate(self.configs):
            # we resolve the path
            config = config.resolve()
            # we get the complete suffix
            config_suffix = ''.join(config.suffixes)
            # we generate a new name using the index followed by the suffix
            config_name = config.with_suffix(
                    f".{index}.{config_suffix}"
            ).name

            shutil.copy2(config, complete_dest_dir / config_name)
