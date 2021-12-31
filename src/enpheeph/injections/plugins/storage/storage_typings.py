# -*- coding: utf-8 -*-
import datetime
import typing


# NOTE: we use typing.Protocol as it is quite difficult to make abc.ABC work with
# SQLAlchemy, so in this way it is easier to use for the different
@typing.runtime_checkable
class ExperimentRunProtocol(typing.Protocol):
    id_: int
    running: bool
    completed: bool
    start_time: typing.Optional[datetime.datetime]
    total_duration: typing.Optional[datetime.timedelta]
    golden_run_flag: bool
    metrics: typing.Optional[typing.Dict[str, typing.Any]]

    polymorphic_discriminator: typing.Optional[str]

    injections: typing.Optional[typing.Sequence["InjectionProtocol"]]

    golden_run: typing.Optional["ExperimentRunProtocol"]
    golden_run_id: typing.Optional[int]

    injected_runs: typing.Optional[typing.Sequence["ExperimentRunProtocol"]]


@typing.runtime_checkable
class InjectionProtocol(typing.Protocol):
    location: typing.Any

    internal_id: int

    experiment_run_id: typing.Optional[int]
    experiment_run: typing.Optional["ExperimentRunProtocol"]


@typing.runtime_checkable
class FaultProtocol(InjectionProtocol, typing.Protocol):
    pass


@typing.runtime_checkable
class MonitorProtocol(InjectionProtocol, typing.Protocol):
    payload: typing.Optional[typing.Dict[str, typing.Any]]
