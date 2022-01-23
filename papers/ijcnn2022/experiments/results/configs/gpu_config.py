# -*- coding: utf-8 -*-
import typing


def config(
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    return {
        "model": {
            "callable_args": {
                "map": "cuda",
            }
        },
        "trainer": {
            "callable_args": {
                "accelerator": "gpu",
                "devices": 1,
            },
        },
    }
