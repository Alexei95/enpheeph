# -*- coding: utf-8 -*-
import abc
import enum
import typing

import enpheeph.handlers.plugins.libraryhandlerpluginabc
import enpheeph.injections.injectionabc
import enpheeph.utils.typings


class HandlerStatus(enum.Enum):
    Running = enum.auto()
    Idle = enum.auto()


class InjectionHandler(object):
    active_injections: typing.List[enpheeph.injections.injectionabc.InjectionABC]
    injections: typing.List[enpheeph.injections.injectionabc.InjectionABC]
    library_handler_plugin: (
        enpheeph.handlers.plugins.libraryhandlerpluginabc.LibraryHandlerPluginABC
    )
    status: HandlerStatus

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

        self.status = HandlerStatus.Idle

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

    def check_running_status(self):
        return self.status == self.status.Running

    def lock_running_status(self):
        if self.check_running_status():
            raise RuntimeError(
                "This function shouldn't have been called " "with a running execution"
            )
        self.status = self.status.Running

    def unlock_running_status(self):
        if not self.check_running_status():
            raise RuntimeError("Handler should have been running")
        self.status = self.status.Idle

    # if None for the arguments, we will activate all the faults
    def activate(
        self,
        injections: typing.Optional[
            typing.Sequence[enpheeph.injections.injectionabc.InjectionABC]
        ] = None,
    ):
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return

        if injections is None:
            injections = self.injections

        self.active_injections = [inj for inj in injections if inj in self.injections]

    # if None we will deactivate everything
    def deactivate(
        self,
        injections: typing.Optional[
            typing.Sequence[enpheeph.injections.injectionabc.InjectionABC]
        ] = None,
    ):
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return

        if injections is None:
            injections = self.injections

        self.active_injections = [
            inj
            for inj in self.active_injections
            if inj not in injections and inj in self.injections
        ]
