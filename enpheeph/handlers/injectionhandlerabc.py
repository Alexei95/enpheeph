import abc
import enum
import typing

import enpheeph.faults.faultabc
import enpheeph.injections.injectionabc
import enpheeph.monitors.monitorabc


class HandlerStatus(enum.Enum):
    Running = enum.auto()
    Idle = enum.auto()


class InjectionHandlerABC(abc.ABC):
    def on_setup_start(self, *args, **kwargs):
        return NotImplemented

    def on_setup_end(self, output, *args, **kwargs):
        return NotImplemented

    def on_teardown_start(self, *args, **kwargs):
        return NotImplemented

    def on_teardown_end(self, output, *args, **kwargs):
        return NotImplemented

    def setup(self, *args, **kwargs):
        self.lock_running_status()
        output = self.on_setup_start(*args, **kwargs)
        output = self.on_setup_end(output, *args, **kwargs)
        return output

    def teardown(self, *args, **kwargs):
        self.unlock_running_status()
        output = self.on_teardown_start(*args, **kwargs)
        output = self.on_teardown_end(output, *args, **kwargs)
        return output

    def __init__(
            self,
            faults: typing.Sequence[enpheeph.faults.faultabc.FaultABC],
            monitors: typing.Sequence[enpheeph.monitors.monitorabc.MonitorABC],
    ):
        self.faults = faults
        self.monitors = monitors

        self.status = HandlerStatus.Idle

        self.active_faults : typing.List[
                enpheeph.faults.faultabc.FaultABC
        ] = []
        self.active_monitors: typing.List[
                enpheeph.monitors.monitorabc.MonitorABC
        ] = []

    def check_running_status(self):
        return self.status == self.status.Running

    def lock_running_status(self):
        if not self.check_running_status():
            self.status = self.status.Running
        else:
            raise RuntimeError("This function shouldn't have been called with a running execution")

    def unlock_running_status(self):
        self.status = self.status.Idle

    @property
    def injection_list(self):
        return tuple(self.faults) + tuple(self.monitors)

    # if None for the arguments, we will activate all the faults
    def activate(
            self,
            faults: typing.Optional[
                    typing.Sequence[enpheeph.faults.faultabc.FaultABC]
            ] = None,
            monitors: typing.Optional[
                    typing.Sequence[enpheeph.monitors.monitorabc.MonitorABC]
            ] = None,
    ):
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return

        if faults is None:
            faults = self.faults

        self.active_faults = [
                f
                for f in faults
                if f in self.faults
        ]

        if monitors is None:
            monitors = self.monitors

        self.active_monitors = [
                m
                for m in monitors
                if m in self.monitors
        ]

    # if None we will deactivate everything
    def deactivate(
            self,
            faults: typing.Optional[
                    typing.Sequence[enpheeph.faults.faultabc.FaultABC]
            ] = None,
            monitors: typing.Optional[
                    typing.Sequence[enpheeph.monitors.monitorabc.MonitorABC]
            ] = None,
    ):
        if self.check_running_status():
            print("Cannot do anything while running, try after the execution")
            return

        if faults is None:
            faults = self.faults

        self.active_faults = [
                f
                for f in self.active_faults
                if f not in faults and f in self.monitors
        ]

        if monitors is None:
            monitors = self.monitors

        self.active_monitors = [
                m
                for m in self.active_monitors
                if m not in monitors and m in self.monitors
        ]
