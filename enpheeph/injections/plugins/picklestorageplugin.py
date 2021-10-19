import copy
import pathlib
import pickle
import typing

import enpheeph.injections.plugins.storagepluginabc
import enpheeph.utils.typings


# we use pickle as short-term temporary storage, not meant for a full 
# experiment
class PickleStoragePlugin(
        enpheeph.injections.plugins.storagepluginabc.StoragePluginABC,
):
    def __init__(
            self,
            path: enpheeph.utils.typings.PathType,
    ):
        self.path = pathlib.Path(path)

        self.list_of_dicts = []
        self.current_dict = {}

    def add_element(self, element_name: str, element: typing.Any) -> None:
        self.current_dict[element_name] = copy.deepcopy(element)

    def add_dict(self, dict_: typing.Dict[str, typing.Any]) -> None:
        self.current_dict.update(
                {
                        key: copy.deepcopy(value)
                        for key, value in dict_.items()
                }
        )

    def submit_eol(self) -> None:
        if self.current_dict:
            self.list_of_dicts.append(self.current_dict)

            self.current_dict = {}

    def execute(self) -> None:
        if not any(self.list_of_dicts):
            return

        with self.path.open('wb') as file:
            pickle.dump(self.list_of_dicts, file)
