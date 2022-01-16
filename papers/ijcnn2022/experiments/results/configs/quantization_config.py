# -*- coding: utf-8 -*-
import typing

import torch
import torch.quantization


def config() -> typing.Dict[str, typing.Any]:
    return {
        "dynamic_quantization_config": {
            "qconfig": {
                torch.nn.Linear,
                torch.nn.LSTM,
                torch.nn.GRU,
                torch.nn.LSTMCell,
                torch.nn.RNNCell,
                torch.nn.GRUCell,
                torch.nn.EmbeddingBag,
            },
            "qdtype": torch.qint8,
        }
    }
