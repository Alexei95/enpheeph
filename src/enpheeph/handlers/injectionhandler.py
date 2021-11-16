# -*- coding: utf-8 -*-
import typing

import enpheeph.handlers.plugins.libraryhandlerpluginabc
import enpheeph.injections.injectionabc
import enpheeph.utils.enums
import enpheeph.utils.typings


class InjectionHandler(object):
    active_injections: typing.List[enpheeph.injections.injectionabc.InjectionABC]
    injections: typing.List[enpheeph.injections.injectionabc.InjectionABC]
    library_handler_plugin: (
        enpheeph.handlers.plugins.libraryhandlerpluginabc.LibraryHandlerPluginABC
    )
    status: enpheeph.utils.enums.HandlerStatus

    def __init__(
        self,
        injections: typing.List[enpheeph.injections.injectionabc.InjectionABC],
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
            typing.Sequence[enpheeph.injections.injectionabc.InjectionABC]
        ] = None,
    ) -> typing.List[enpheeph.injections.injectionabc.InjectionABC]:
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return self.active_injections

        if injections is None:
            injections = self.injections

        self.active_injections = [inj for inj in injections if inj in self.injections]

        return self.active_injections

    # if None we will deactivate everything
    # it returns the active injections
    def deactivate(
        self,
        injections: typing.Optional[
            typing.Sequence[enpheeph.injections.injectionabc.InjectionABC]
        ] = None,
    ) -> typing.Sequence[enpheeph.injections.injectionabc.InjectionABC]:
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
