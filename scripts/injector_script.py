# -*- coding: utf-8 -*-
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

import pathlib

import enpheeph


def get_cifar10_classfier_config():
    return {}


def main() -> None:
    storage_file = (
        pathlib.Path(__file__).absolute().parent.parent
        / "results/injection_test/database.sqlite"
    )
    storage_file.parent.mkdir(exist_ok=True, parents=True)

    pytorch_handler_plugin = enpheeph.handlers.plugins.PyTorchHandlerPlugin()
    storage_plugin = enpheeph.injections.plugins.storage.SQLiteStoragePlugin(
        db_url="sqlite:///" + str(storage_file)
    )

    injection_handler = enpheeph.handlers.InjectionHandler(
        injections=[],
        library_handler_plugin=pytorch_handler_plugin,
    )

    # we delay the instantiation of the callback to allow the saving of the
    # current configuration
    callback = enpheeph.integrations.pytorchlightning.InjectionCallback(
        injection_handler=injection_handler,
        storage_plugin=storage_plugin,
        # this config used to contain the complete system config: trainer + model +
        # dataset, including the configuration for injections
        # extra_session_info=config,
    )


if __name__ == "__main__":
    main()
