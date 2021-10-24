# -*- coding: utf-8 -*-
import typing

import sqlalchemy
import sqlalchemy.dialects.postgresql
import sqlalchemy.ext.compiler
import sqlalchemy.orm
import sqlalchemy.sql.expression
import sqlalchemy.types

import enpheeph.injections.plugins.sqlstorageplugin.sqlutils
import enpheeph.utils.enums


# we define the metadata with the registry and the base class to identify
# rows in tables
Base: sqlalchemy.orm.decl_api.DeclarativeMeta = sqlalchemy.orm.declarative_base()


# we define all the classes, each one represents a row entry

# the initial idea was to use as much SQL as possible, but this might
# become cumbersome in complex datatypes, like injections and results,
# where multidimensionality is high and types can be of different base type


# experiment_run contains all the information of each single run
# there are connections to the injections in a many-to-many direction
# additionally, each run corresponds to a result
class ExperimentRun(Base):
    __tablename__ = "experiment_run"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    # golden_run = sqlalchemy.Column(sqlalchemy.Bool, nullable=False)
    # maybe use a golden_run_referral to refer to the proper golden_run
    running = sqlalchemy.Column(sqlalchemy.Bool, nullable=False)
    completed = sqlalchemy.Column(sqlalchemy.Bool, nullable=False)
    start_time = sqlalchemy.Column(sqlalchemy.Time)
    total_duration = sqlalchemy.Column(sqlalchemy.Time)

    # in this case we have a many-to-many relationship
    injections = sqlalchemy.relationship(
        "Injection",
        secondary=lambda: experiment_injection_association_table,
        back_populates="experiment_runs",
    )

    # here for each run we can have a single result, so we have
    # a one-to-one
    # to have a one-to-one we do a one-to-many with relationship and
    # a many-to-one again with relationship
    # additionally the one-to-many becomes a single scalar instead of a list
    # additionally, one may enforce only one element in the many-to-one
    # on the one side, by forcing unique=True
    experiment_result = sqlalchemy.relationship(
        "ExperimentResult", back_populates="experiment_run", uselist=False,
    )

    golden_run_flag = sqlalchemy.Column(sqlalchemy.Boolean)

    # we generate a one-to-many relationship from the faulty runs
    # back to the golden reference run
    faulty_runs = sqlalchemy.relationship(
        "ExperimentRun", back_populates="golden_run_reference",
    )

    golden_run_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey("experiment_run.id"),
    )
    golden_run_reference = sqlalchemy.relationship(
        "ExperimentRun", back_populates="faulty_runs",
    )


class Injection(Base):
    __tablename__ = "injection"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    experiment_runs = sqlalchemy.relationship(
        "ExperimentRun",
        secondary=lambda: experiment_injection_association_table,
        back_populates="injections",
    )
    tensor_index = sqlalchemy.Column(sqlalchemy.PickleType())
    bit_index = sqlalchemy.Column(sqlalchemy.PickleType())
    time_index = sqlalchemy.Column(sqlalchemy.PickleType())
    module_name = sqlalchemy.Column(sqlalchemy.String())
    parameter_type = sqlalchemy.Column(
        sqlalchemy.Enum(enpheeph.utils.enums.ParameterType)
    )

    # for fault, one-to-one with a fault table


class Fault(Base):
    __tablename__ = "fault"


class ExperimentResult(Base):
    __tablename__ = "experiment_result"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    experiment_run = sqlalchemy.relationship(
        "ExperimentRun", back_populates="experiment_result",
    )
    experiment_run_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey("experiment_run.id"), unique=True,
    )

    average_test_accuracy = sqlalchemy.Column(sqlalchemy.Float)
    average_test_loss = sqlalchemy.Column(sqlalchemy.Float)


class ExperimentLogging(Base):
    __tablename__ = "experiment_logging"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    module_name = sqlalchemy.Column(sqlalchemy.String)
    parameter_type = sqlalchemy.Column(
        sqlalchemy.Enum(
            enpheeph.utils.enums.ParameterType,
            validate_strings=True,
            omit_aliases=False,
        ),
    )
    metric_names = sqlalchemy.Column(sqlalchemy.ARRAY(sqlalchemy.String))
    metric_values = sqlalchemy.Column(sqlalchemy.ARRAY(sqlalchemy.Float))


# we define this association table to allow for multiple runs to refer to
# the same injection and vice versa
experiment_injection_association_table: sqlalchemy.Table = sqlalchemy.Table(
    "experimentrun_injection_association",
    Base.metadata,
    sqlalchemy.Column("experiment_run", sqlalchemy.ForeignKey("experiment_run.id"),),
    sqlalchemy.Column("injection", sqlalchemy.ForeignKey("injection.id"),),
)


import dataclasses

import enpheeph.utils.data_classes
import enpheeph.utils.enums
import enpheeph.utils.typings


# redefine the fields using the declarative example
# redefine the tables using dataclasses where required
# consider adding a mixin for importing the fields from the parent class
# e.g. from_parent_instance(FaultLocation(...))
# just to copy the fields inside and use the sqlalchemy magic
@dataclasses.dataclass
class SQLFaultLocation(enpheeph.utils.data_classes.FaultLocation):
    # name of the module to be targeted
    module_name: str
    # type of parameter, activation or weight
    parameter_type: enpheeph.utils.enums.ParameterType
    # tensor index which can be represented using a numpy/pytorch indexing
    # array
    tensor_index: enpheeph.utils.typings.IndexType
    # same for the bit injection info
    bit_index: enpheeph.utils.typings.BitIndexType

    bit_fault_value: enpheeph.utils.enums.BitFaultValue

    time_index: (
        typing.Optional[enpheeph.utils.typings.TimeIndexType]
    ) = dataclasses.field(default=None)
