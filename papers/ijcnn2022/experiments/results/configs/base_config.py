# -*- coding: utf-8 -*-
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
                "callbacks": [],
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
