import copy
import os
import pathlib
import shutil
import sys
import time
import typing

import pytorch_lightning.callbacks

import src.utils.functions


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
    }
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
            version: typing.Union[int, bool] = True,
            time_info: bool = True,
    ):
        super().__init__()

        self.configs = [pathlib.Path(c) for c in configs]
        self.dest_dir = pathlib.Path(dest_dir)
        self.subdirectory = subdirectory
        self.version = version
        self.time_info = time_info

    def make_config(self):
        self.CONFIG_PATHS = {
                index: str(c.resolve())
                for index, c in enumerate(self.configs)
        }
        self.KWARGS_DICT['dest_dir'] = str(self.dest_dir.resolve())
        self.KWARGS_DICT['subdirectory'] = self.subdirectory

        return self.DICT_CONFIG

    def on_train_start(
            self,
            trainer,
            pl_module
    ) -> None:
        # we generate the complete destination file
        # and we make sure it exists
        complete_dest_dir = self.dest_dir.resolve() / self.subdirectory
        # we instantiate the list with the extra information for the
        # subdirectory
        extra_info_subdir = []
        # version checks
        if self.version is True:
            versions = complete_dest_dir.glob('**')
            # add more checks here for 0 and for other possible files
            # add flag for inserting time info

            # to convert the string name to integer
            # we assume it is in the format (version)_[versionnumber]_time_info
            # so we are interested in the second element
            def converter(x):
                return int(x.split('_')[1])

            # we convert the list of file names into a list of integers
            # if one of them cannot be converted, we skip it
            converted_versions = []
            for v in versions:
                # we only check for directories
                if v.is_dir():
                    try:
                        converted_versions.append(converter(v.name))
                    except (ValueError, IndexError):
                        pass

            # we sort the remaining numbers
            sorted_versions = sorted(converted_versions)
            # if there are some, we get the last one and increase it,
            # otherwise we use 0 if empty
            if not len(sorted_versions):
                next_version = 0
            else:
                next_version = sorted_versions[-1] + 1

            extra_info_subdir.extend(['version', str(next_version)])
        # if it is a number we use it directly
        elif isinstance(self.version, int):
            extra_info_subdir.append(str(self.version))

        # time checks
        # if we should add time info, we get the current UTC time
        if self.time_info:
            time_info = src.utils.functions.current_utctime_string()
            extra_info_subdir.append(time_info)
        # here we join the extra sub-components
        # if the list is empty, it will return the same directory when parsed
        # with pathlib
        complete_dest_dir = complete_dest_dir / '_'.join(extra_info_subdir)
        complete_dest_dir.mkdir(exist_ok=True, parents=True)
        # we cycle through the configurations
        for index, config in enumerate(self.configs):
            # we resolve the path
            config = config.resolve()
            # we get the complete suffix
            config_suffix = ''.join(config.suffixes)
            # we generate a new name using the index followed by the suffix
            # we don't need a dot in between as the first suffix
            # has already one inside
            config_name = config.with_suffix(
                    f".{index}{config_suffix}"
            ).name

            shutil.copy2(config, complete_dest_dir / config_name)
