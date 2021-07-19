import copy
import functools
import pathlib

import typing

import pl_bolts
import pytorch_lightning
import tonic.datasets.nmnist
import torch

import src.ml.pl.datasets.utils.monkeypatchedtonicdataset


class NMNISTDataModule(
        pl_bolts.datamodules.vision_datamodule.VisionDataModule
):

    EXTRA_ARGS = {'first_saccade_only': False}
    dataset_cls = src.ml.pl.datasets.utils.\
        monkeypatchedtonicdataset.monkey_patching_tonic_dataset(
                tonic.datasets.nmnist.NMNIST,
                tonic.datasets.nmnist
        )
    name = "nmnist"
    dims = (1, 34, 34)

    def __init__(
            self,
            # generic VisionDataModule arguments
            data_dir: typing.Optional[str] = None,
            val_split: typing.Union[int, float] = 0.2,
            num_workers: int = 16,
            normalize: bool = False,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = False,
            pin_memory: bool = False,
            drop_last: bool = False,
            # generic transforms
            train_transforms: typing.Optional[typing.Callable] = None,
            val_transforms: typing.Optional[typing.Callable] = None,
            test_transforms: typing.Optional[typing.Callable] = None,
            # tonic specific arguments for collate_fn and target transform
            target_transform: typing.Optional[typing.Callable] = None,
            collate_fn: typing.Optional[typing.Callable] = None,
            # N-MNIST specific arguments
            first_saccade_only: bool = False,
            # extra argument
            *args: typing.Any,
            **kwargs: typing.Any,
    ):
        super().__init__(
                *args,
                data_dir=data_dir,
                val_split=val_split,
                num_workers=num_workers,
                normalize=normalize,
                batch_size=batch_size,
                seed=seed,
                shuffle=shuffle,
                pin_memory=pin_memory,
                drop_last=drop_last,
                **kwargs)

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms

        self.target_transform = target_transform
        # to pass the custom target transform
        self.dataset_cls = functools.partial(
                self.dataset_cls,
                target_transform=self.target_transform
        )
        self.collate_fn = collate_fn

        self.first_saccade_only = first_saccade_only

        # this is automatically passed in the dataset class
        self.EXTRA_ARGS['first_saccade_only'] = self.first_saccade_only

    def default_transforms(self) -> typing.Callable:
        return tonic.transforms.Compose()

    def _data_loader(
            self,
            dataset: torch.utils.data.Dataset,
            shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
                pin_memory=self.pin_memory
        )
