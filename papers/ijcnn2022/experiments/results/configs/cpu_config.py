# -*- coding: utf-8 -*-
import typing


def config(
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    return {
        "model": {
            "callable_args": {
                "map": "cpu",
            }
        },
        "trainer": {
            "callable_args": {
                "accelerator": "cpu",
                "devices": 1,
            },
        },
    }
