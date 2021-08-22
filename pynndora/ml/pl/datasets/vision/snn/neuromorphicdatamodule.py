import abc
import copy
import functools
import pathlib

import typing

import pl_bolts
import pytorch_lightning
import torch


class NeuromorphicDataModule(
        abc.ABC,
        pl_bolts.datamodules.vision_datamodule.VisionDataModule
):
    EXTRA_ARGS = {'target_transform': None}
    dataset_cls = type
    name = None
    dims = tuple()

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
                **kwargs
        )

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_transforms = val_transforms

        self.target_transform = target_transform
        self.collate_fn = collate_fn

        # this is automatically passed in the dataset class
        self.EXTRA_ARGS['target_transform'] = self.target_transform

    @abc.abstractmethod
    def default_transforms(self) -> typing.Callable:
        pass

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
