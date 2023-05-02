# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2023 Alessio "Alexei95" Colucci
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

import pathlib
import typing

import flash
import flash.image

import enpheeph
import enpheeph.injections.plugins.mask.autopytorchmaskplugin

import base_config
import cifar10_config
import quantization_config


def config(
    *,
    dataset_directory: pathlib.Path,
    model_weight_file: pathlib.Path,
    storage_file: pathlib.Path,
    **kwargs: typing.Any,
) -> typing.Dict[str, typing.Any]:
    pytorch_handler_plugin = enpheeph.handlers.plugins.PyTorchHandlerPlugin()
    storage_plugin = enpheeph.injections.plugins.storage.SQLiteStoragePlugin(
        db_url="sqlite:///" + str(storage_file)
    )

    injection_handler = enpheeph.handlers.InjectionHandler(
        injections=[],
        library_handler_plugin=pytorch_handler_plugin,
    )

    model = {
        "callable": flash.image.ImageClassifier.load_from_checkpoint,
        "callable_args": {
            "checkpoint_path": str(model_weight_file),
            # issues with loading GPU model on CPU
            # it should work with PyTorch but there must be some problems with
            # PyTorch Lightning/Flash leading to use some GPU memory
            "map_location": "cpu",
        },
    }

    config = base_config.config()
    # datamodule update
    config.update(cifar10_config.config(dataset_directory=dataset_directory))
    # dynamic quantization update
    config.update(quantization_config.config())
    config["model"] = model
    # update the Trainer with flash as we are using flash models, to avoid
    # compatibility issues such as CUDA out of memory on CPU-only
    config["trainer"]["callable"] = flash.Trainer

    # we delay the instantiation of the callback to allow the saving of the
    # current configuration
    callback = enpheeph.integrations.pytorchlightning.InjectionCallback(
        injection_handler=injection_handler,
        storage_plugin=storage_plugin,
        extra_session_info=config,
    )

    config["trainer"]["callable_args"]["callbacks"].append(callback)

    # to save the injection handler to enable/disable faults
    config["injection_handler"] = injection_handler
    # to save the callback to access to the same storage plugin
    config["injection_callback"] = callback

    # custom is used to avoid the random injections
    config["injection_config"] = {}

    return config
