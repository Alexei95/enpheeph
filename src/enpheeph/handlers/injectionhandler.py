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

import enpheeph.handlers.plugins.libraryhandlerpluginabc
import enpheeph.injections.abc.injectionabc
import enpheeph.utils.enums
import enpheeph.utils.typings


class InjectionHandler(object):
    active_injections: typing.List[enpheeph.injections.abc.injectionabc.InjectionABC]
    injections: typing.List[enpheeph.injections.abc.injectionabc.InjectionABC]
    library_handler_plugin: (
        enpheeph.handlers.plugins.libraryhandlerpluginabc.LibraryHandlerPluginABC
    )
    status: enpheeph.utils.enums.HandlerStatus

    def __init__(
        self,
        injections: typing.List[enpheeph.injections.abc.injectionabc.InjectionABC],
        library_handler_plugin: (
            enpheeph.handlers.plugins.libraryhandlerpluginabc.LibraryHandlerPluginABC
        ),
    ):
        self.injections = list(injections)
        self.library_handler_plugin = library_handler_plugin

        self.active_injections = []

        self.status = enpheeph.utils.enums.HandlerStatus.Idle

    def setup(
        self, model: enpheeph.utils.typings.ModelType
    ) -> enpheeph.utils.typings.ModelType:
        self.lock_running_status()
        model = self.library_handler_plugin.library_setup(model, self.active_injections)
        return model

    def teardown(
        self, model: enpheeph.utils.typings.ModelType
    ) -> enpheeph.utils.typings.ModelType:
        model = self.library_handler_plugin.library_teardown(
            model, self.active_injections
        )
        self.unlock_running_status()
        return model

    def check_running_status(self) -> bool:
        # mypy has errors with enums, might be fixed using py.typed
        return self.status == self.status.Running  # type: ignore[comparison-overlap]

    def lock_running_status(self) -> bool:
        if self.check_running_status():
            raise RuntimeError(
                "This function shouldn't have been called " "with a running execution"
            )
        # mypy has errors with enums, might be fixed using py.typed
        self.status = self.status.Running  # type: ignore[assignment]
        # we return True if the operation is successful
        return True

    def unlock_running_status(self) -> bool:
        if not self.check_running_status():
            raise RuntimeError("Handler should have been running")
        # mypy has errors with enums, might be fixed using py.typed
        self.status = self.status.Idle  # type: ignore[assignment]
        # we return True if the operation is successful
        return True

    # if None for the arguments, we will activate all the faults
    # it returns the active injections
    def activate(
        self,
        injections: typing.Optional[
            typing.Sequence[enpheeph.injections.abc.injectionabc.InjectionABC]
        ] = None,
    ) -> typing.List[enpheeph.injections.abc.injectionabc.InjectionABC]:
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return self.active_injections

        if injections is None:
            injections = self.injections

        # we use a dict to filter the duplicates in injections + self.active_injections
        # otherwise bad things might happen in the SQL as the same object will be
        # processed multiple times
        filtered_injections = {
            inj: counter
            for counter, inj in enumerate(list(injections) + self.active_injections)
        }.keys()

        self.active_injections = [
            inj for inj in filtered_injections if inj in self.injections
        ]

        return self.active_injections

    # if None we will deactivate everything
    # it returns the active injections
    def deactivate(
        self,
        # here Sequence is fine as we are simply iterating over/checking presence
        injections: typing.Optional[
            typing.Sequence[enpheeph.injections.abc.injectionabc.InjectionABC]
        ] = None,
    ) -> typing.Sequence[enpheeph.injections.abc.injectionabc.InjectionABC]:
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return self.active_injections

        if injections is None:
            injections = self.injections

        self.active_injections = [
            inj
            for inj in self.active_injections
            if inj not in injections and inj in self.injections
        ]

        return self.active_injections

    # to add injections to the current list of injections
    def add_injections(
        self,
        injections: typing.Sequence[enpheeph.injections.abc.injectionabc.InjectionABC],
    ) -> typing.Sequence[enpheeph.injections.abc.injectionabc.InjectionABC]:
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return self.injections

        # we use a dict to filter the duplicates in injections + self.active_injections
        # otherwise bad things might happen in the SQL as the same object will be
        # processed multiple times
        filtered_injections = {
            inj: counter
            for counter, inj in enumerate(list(injections) + self.injections)
        }.keys()

        self.injections = list(filtered_injections)

        # we call activate with the list of active injections to remove
        # the ones not included
        self.activate(self.active_injections)

        return self.injections

    # to remove injections from the current list
    # if None we remove all of them
    def remove_injections(
        self,
        injections: typing.Optional[
            typing.Sequence[enpheeph.injections.abc.injectionabc.InjectionABC]
        ] = None,
    ) -> typing.Sequence[enpheeph.injections.abc.injectionabc.InjectionABC]:
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return self.injections

        if injections is None:
            injections = self.injections

        self.injections = [inj for inj in self.injections if inj not in injections]

        # we call activate with the list of active injections to remove
        # the ones not included
        self.activate(self.active_injections)

        return self.injections
