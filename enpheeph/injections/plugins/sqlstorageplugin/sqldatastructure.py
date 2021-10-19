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
Base: sqlalchemy.orm.decl_api.DeclarativeMeta = (
    sqlalchemy.orm.declarative_base()
)


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
    # reference to the golden run to which the results of this run have to
    # be compared
    # this is a one-to-one
    golden_run_reference = sqlalchemy.relationship(
        "GoldenRun", back_populates="injected_runs",
    )
    golden_run_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey("golden_run.id")
    )


class GoldenRun(Base):
    __tablename__ = "golden_run"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    # in this way each entry here is mapped onto a single entry in
    # experiment_run
    # NOTE: check how to avoid repetitions -> might be done with unique=True
    experiment_run = sqlalchemy.Column(
        sqlalchemy.Integer,
        sqlalchemy.ForeignKey("experiment_run.id"),
        # to avoid repetitions
        unique=True,
    )
    # instead for the injected runs we want a one-to-many, each golden run
    # can be mapped to many injected runs
    injected_runs = sqlalchemy.relationship(
        "ExperimentRun", back_populates="golden_run_reference",
    )


class Injection(Base):
    __tablename__ = "injection"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    experiment_runs = sqlalchemy.relationship(
        "ExperimentRun",
        secondary=lambda: experiment_injection_association_table,
        back_populates="injections",
    )
    tensor_index = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)

    


class ExperimentResult(Base):
    __tablename__ = "experiment_result"

    id = sqlalchemy.Column(sqlalchemy.Integer, primary_key=True)
    experiment_run = sqlalchemy.relationship(
        "ExperimentRun", back_populates="experiment_result",
    )
    experiment_run_id = sqlalchemy.Column(
        sqlalchemy.Integer, sqlalchemy.ForeignKey("experiment_run.id")
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
    sqlalchemy.Column(
        "experiment_run", sqlalchemy.ForeignKey("experiment_run.id"),
    ),
    sqlalchemy.Column("injection", sqlalchemy.ForeignKey("injection.id"),),
)
