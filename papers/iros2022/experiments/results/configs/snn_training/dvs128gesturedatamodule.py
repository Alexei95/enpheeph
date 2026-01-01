# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2026 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2022 Alessio "Alexei95" Colucci
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import typing

import pl_bolts
import tonic
import torch
import torchvision


class DVS128GestureDataModule(
    pl_bolts.datamodules.vision_datamodule.VisionDataModule,
):
    DEFAULT_TRAIN_TRANSFORMS = tonic.transforms.Compose(
        [
            # torch.tensor,
            # tonic.transforms.Downsample(time_factor=0.0001),
            # average number of timesteps is 7185841
            # so we can use a time window of 100000 to make it into 72
            tonic.transforms.MergePolarities(),
            tonic.transforms.ToFrame(
                tonic.datasets.dvsgesture.DVSGesture.sensor_size,
                time_window=25_000,
            ),
        ]
    )
    DEFAULT_VAL_TRANSFORMS = DEFAULT_TRAIN_TRANSFORMS
    DEFAULT_TEST_TRANSFORMS = DEFAULT_TRAIN_TRANSFORMS

    DEFAULT_TARGET_TRANSFORM = None
    DEFAULT_COLLATE_FN = torchvision.transforms.Compose(
        [
            tonic.collation.PadTensors(batch_first=True),
        ]
    )

    EXTRA_ARGS = {"target_transform": None}

    # trick as dataset_cls should have this signature, using also download which is
    # not required in tonic
    # see the corresponding property
    # dataset_cls = tonic.datasets.dvsgesture.DVSGesture
    name = "DVSGesture"
    dims = tonic.datasets.dvsgesture.DVSGesture.sensor_size
    num_classes = 11

    # trick as dataset_cls should have the signature of dataset_cls_interface,
    # using also download which is not used in tonic
    @property
    def dataset_cls(self):
        def dataset_cls_interface(
            data_dir, train=True, download=True, transform=None, *args, **kwargs
        ):
            return tonic.datasets.dvsgesture.DVSGesture(
                save_to=data_dir, train=train, transform=transform
            )

        return dataset_cls_interface

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
        train_transforms: typing.Optional[
            typing.Callable[[typing.Any], torch.Tensor]
        ] = None,
        val_transforms: typing.Optional[
            typing.Callable[[typing.Any], torch.Tensor]
        ] = None,
        test_transforms: typing.Optional[
            typing.Callable[[typing.Any], torch.Tensor]
        ] = None,
        # tonic specific arguments for collate_fn and target transform
        target_transform: typing.Optional[
            typing.Callable[[typing.Any], torch.Tensor]
        ] = None,
        collate_fn: typing.Optional[
            typing.Callable[[torch.Tensor], torch.Tensor]
        ] = None,
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
            **kwargs,
        )

        if train_transforms is None:
            self.train_transforms = self.DEFAULT_TRAIN_TRANSFORMS
        else:
            self.train_transforms = train_transforms
        if val_transforms is None:
            self.val_transforms = self.DEFAULT_VAL_TRANSFORMS
        else:
            self.val_transforms = val_transforms
        if test_transforms is None:
            self.test_transforms = self.DEFAULT_TEST_TRANSFORMS
        else:
            self.test_transforms = test_transforms

        if target_transform is None:
            self.target_transform = self.DEFAULT_TARGET_TRANSFORM
        else:
            self.target_transform = target_transform
        if collate_fn is None:
            self.collate_fn = self.DEFAULT_COLLATE_FN
        else:
            self.collate_fn = collate_fn

        # this is automatically passed in the dataset class
        self.EXTRA_ARGS["target_transform"] = self.target_transform

        # we call it here to initialize the datasets otherwise when using *_dataloader
        # it is not automatically called
        self.setup()

    def default_transforms(self) -> typing.Callable[[typing.Any], torch.Tensor]:
        return tonic.transforms.Compose([])

    def _data_loader(
        self, dataset: torch.utils.data.Dataset, shuffle: bool = False
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
