import copy
import pathlib
import typing

# this decorator is useful for monkey-patching the dataset
def monkey_patching_tonic_dataset(dataset_class, dataset_module):
    class MonkeyPatchedTonicDataset(dataset_class):
        def check_integrity(
                self,
                fpath: str,
                md5: typing.Optional[str] = None
        ) -> bool:
            if self._check_dataset_existence():
                # FIXME: improve this message
                print(
                        'Verified existing directory structure, '
                        'skipping file checksum check'
                )
                return True
            else:
                return self._original_check_integrity(fpath, md5)

        def __init__(self, *args, **kwargs):
            self._original_check_integrity = copy.deepcopy(
                    dataset_module.check_integrity
            )
            dataset_module.check_integrity = self.check_integrity
            super().__init__(*args, **kwargs)

        # with this function we check the existence of the directories for the
        # testing and training subsets, by also checking the high-level
        # directory structure to match with the expected one
        def _check_dataset_existence(self):
            for file_ in (self.test_filename, self.train_filename):
                path_file = pathlib.Path(self.location_on_system) / file_
                path_dir = path_file.with_suffix('')

                expected_dir_list = {str(i) for i in range(len(self.classes))}
                dir_list = set(path_dir.glob('*'))

                if not path_dir.is_dir() or expected_dir_list.issubset(dir_list):
                    return False
            return True

        # this function is called only if the download flag is True in the init
        def download(self):
            if self._check_dataset_existence():
                # FIXME: improve this message
                print(
                        'Verified existing directory structure, '
                        'skipping download and extraction'
                )
                return

            super().download()

        def __del__(self):
            dataset_module.check_integrity = self._original_check_integrity

    return MonkeyPatchedTonicDataset
