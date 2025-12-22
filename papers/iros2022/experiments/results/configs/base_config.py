# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2025 Alessio "Alexei95" Colucci
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

import pytorch_lightning


def config(
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    return {
        "seed_everything": {
            "seed": 42,
            "workers": True,
        },
        "model": {},
        "datamodule": {},
        "injection_handler": {},
        "trainer": {
            "callable": pytorch_lightning.Trainer,
            "callable_args": {
                "callbacks": [
                    pytorch_lightning.callbacks.TQDMProgressBar(
                        refresh_rate=10,
                    )
                ],
                "deterministic": True,
                "enable_checkpointing": False,
                "max_epochs": 1,
                # one can use gpu but some functions will not be deterministic,
                # so deterministic
                # must be set to False
                "accelerator": "gpu",
                "devices": 1,
                # if one uses spawn or dp it will fail
                # as sqlite connector is not picklable
                # "strategy": "ddp",
            },
        },
    }
