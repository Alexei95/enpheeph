# -*- coding: utf-8 -*-
import dataclasses
import datetime
import typing

import sqlalchemy
import sqlalchemy.dialects.postgresql
import sqlalchemy.ext.compiler
import sqlalchemy.ext.mutable
import sqlalchemy.inspection
import sqlalchemy.orm
import sqlalchemy.orm.decl_api
import sqlalchemy.sql.expression
import sqlalchemy.types

import enpheeph.injections.plugins.storage.sqlstorageplugin.sqlutils
import enpheeph.utils.data_classes
import enpheeph.utils.enums
import enpheeph.utils.functions


# this string is used to identify
# the SQLAlchemy metadata in each field of each dataclass
SQLALCHEMY_METADATA_KEY: str = "sqlalchemy"


# we define the metadata with the registry and the base class to identify
# rows in tables
# mapper_registry = sqlalchemy.orm.registry()
# we don't need it if we use only dataclasses
# or sqlalchemy.orm.declarative_base() if we don't use the mapper_registry
# Base: sqlalchemy.orm.decl_api.DeclarativeMeta = mapper_registry.generate_base()
#
# defining our custom base class,
# we can define attributes which are common across the different
# NOTE: the whole assumption here is that we can have inheritance but they **must** be
# connected with a joined table inheritance
@sqlalchemy.orm.declarative_mixin
class CustomBaseClass(object):
    # ClassVar to avoid the field to be considerate in dataclasses
    ID_NAME: typing.ClassVar[str] = "id_"
    # PARENT_CLASS: typing.Type['CustomBaseClass']

    @classmethod
    @property
    def snake_case_class_name(cls) -> str:
        snake_case_name: str = enpheeph.utils.functions.camel_to_snake(cls.__name__)
        return snake_case_name

    # cascading is not applicable to __magic__ attributes
    # however this is called by all classes, even children, unless overwritten
    @sqlalchemy.orm.declared_attr
    def __tablename__(cls) -> sqlalchemy.orm.Mapped[typing.Optional[str]]:
        if sqlalchemy.orm.has_inherited_table(cls):
            # if it is a inherited class, we don't need the tablename as we are using
            # directly the joined table inheritance
            return None
        else:
            return cls.snake_case_class_name

    # NOTE: id is created after an object is committed to the SQL DB
    # we are using no table for the subclasses so we cannot have a primary id
    # for each subclasses
    @sqlalchemy.orm.declared_attr.cascading
    def id_(cls) -> sqlalchemy.orm.Mapped[typing.Optional[int]]:
        # not required, it gives out
        # sqlalchemy.exc.ArgumentError:
        # Can't place primary key columns on an inherited class with no table.
        # ^ this error if __tablename__ is None
        if sqlalchemy.orm.has_inherited_table(cls):
            return None
            # return sqlalchemy.Column(
            #     cls.ID_NAME,
            #     sqlalchemy.ForeignKey(
            #         f"{cls.PARENT_CLASS.__tablename__}.{cls.PARENT_CLASS.ID_NAME}"
            #     ),
            #     primary_key=True
            # )
        else:
            # ID_NAME changes only the column name inside the SQL, at ORM-level is
            # always id_
            return sqlalchemy.Column(cls.ID_NAME, sqlalchemy.Integer, primary_key=True)

    # look for specific args for PostgreSQL
    # __table_args__ = {'mysql_engine': 'InnoDB'}


CustomBase = sqlalchemy.orm.declarative_base(cls=CustomBaseClass)


# NOTE: declarative mixin is only useful for MyPy,
# it does not provide any extra functionality
@sqlalchemy.orm.declarative_mixin
class ExperimentRunBaseMixin(object):
    # NOTE: all of these declared_attr need to be mapped using mapped_registry.mapped
    # or inherit from a Base class
    # then these attributes will become settable in the init of the corresponding
    # class, much like a dataclass
    # NOTE: we use cascading so that the definition propagates also to the children
    @sqlalchemy.orm.declared_attr.cascading
    def running(cls) -> sqlalchemy.orm.Mapped[bool]:
        return sqlalchemy.Column(sqlalchemy.Boolean, nullable=False)

    @sqlalchemy.orm.declared_attr.cascading
    def completed(cls) -> sqlalchemy.orm.Mapped[bool]:
        return sqlalchemy.Column(sqlalchemy.Boolean, nullable=False)

    @sqlalchemy.orm.declared_attr.cascading
    def start_time(cls) -> sqlalchemy.orm.Mapped[typing.Optional[datetime.datetime]]:
        return sqlalchemy.Column(sqlalchemy.DateTime)

    @sqlalchemy.orm.declared_attr.cascading
    def total_duration(cls) -> sqlalchemy.orm.Mapped[typing.Optional[float]]:
        return sqlalchemy.Column(sqlalchemy.Float)

    @sqlalchemy.orm.declared_attr.cascading
    def golden_run_flag(cls) -> sqlalchemy.orm.Mapped[bool]:
        return sqlalchemy.Column(sqlalchemy.Boolean, nullable=False)

    @sqlalchemy.orm.declared_attr.cascading
    def metrics(
        self,
    ) -> sqlalchemy.orm.Mapped[typing.Optional[typing.Dict[str, typing.Any]]]:
        return sqlalchemy.Column(
            sqlalchemy.ext.mutable.MutableDict.as_mutable(sqlalchemy.PickleType)
        )


@sqlalchemy.orm.declarative_mixin
class PolymorphicMixin(object):
    POLYMORPHIC_DISCRIMINATOR_NAME: typing.ClassVar[str] = "polymorphic_discriminator"

    @sqlalchemy.orm.declared_attr
    def __mapper_args__(
        cls: typing.Type[CustomBaseClass],
    ) -> sqlalchemy.orm.Mapped[typing.Dict[str, str]]:
        if sqlalchemy.orm.has_inherited_table(cls):
            # the name is the snake_case name of the class since __tablename__ is not
            # defined for the children classes
            return {
                "polymorphic_identity": cls.snake_case_class_name,
            }
        else:
            # for the parent class we use the tablename as identity
            return {
                "polymorphic_identity": cls.__tablename__,
                "polymorphic_on": cls.POLYMORPHIC_DISCRIMINATOR_NAME,
            }

    # this is defined only for the main class
    @sqlalchemy.orm.declared_attr
    def polymorphic_discriminator(cls) -> sqlalchemy.orm.Mapped[typing.Optional[str]]:
        if sqlalchemy.orm.has_inherited_table(cls):
            return None
        else:
            return sqlalchemy.Column(sqlalchemy.String)


# no need for the dataclass if we are instantiating everything normally and we
# don't need other __magic__ methods from dataclass
@dataclasses.dataclass(init=True, repr=True, eq=True)
class ExperimentRun(ExperimentRunBaseMixin, PolymorphicMixin, CustomBase):
    # FIXME: add support for ModelInfo, which might be a one-to-many from the ModelInfo
    # side

    INJECTION_CLASS_LAMBDA: typing.ClassVar[
        typing.Callable[..., typing.Type["CustomBaseClass"]]
    ] = lambda: Injection
    INJECTION_FOREIGN_KEY_LAMBDA: typing.ClassVar[
        typing.Callable[..., sqlalchemy.Column[sqlalchemy.Integer]]
    ] = lambda: Injection.experiment_run_id
    INJECTION_BACKPOPULATES_NAME: typing.ClassVar[str] = "experiment_run"

    # relationship for having a list of InjectedRun subjected to this GoldenRun
    # we also create golden_run as referral back to the golden run
    # foreign_keys is the golden_run_id containing the ID of the golden run
    # to connect the many remote side with InjectedRun
    # back to the one local side of the golden run
    @sqlalchemy.orm.declared_attr
    def injected_runs(cls) -> sqlalchemy.orm.Mapped[typing.Sequence["ExperimentRun"]]:
        return sqlalchemy.orm.relationship(
            cls.__name__,
            backref=sqlalchemy.orm.backref("golden_run", remote_side=[cls.id_]),
            foreign_keys=f"{cls.__name__}.golden_run_id",
            cascade="all, delete-orphan",
        )

    @sqlalchemy.orm.declared_attr
    def golden_run_id(cls) -> sqlalchemy.orm.Mapped[typing.Optional[int]]:
        return sqlalchemy.Column(
            sqlalchemy.ForeignKey(f"{cls.__tablename__}.{cls.ID_NAME}")
        )

    # a list of all the injections in this experiment
    @sqlalchemy.orm.declared_attr
    def injections(cls) -> sqlalchemy.orm.Mapped[typing.Sequence["Injection"]]:
        return sqlalchemy.orm.relationship(
            cls.INJECTION_CLASS_LAMBDA,
            back_populates=cls.INJECTION_BACKPOPULATES_NAME,
            foreign_keys=cls.INJECTION_FOREIGN_KEY_LAMBDA,
            cascade="all, delete-orphan",
        )


# no need for the dataclass if we are instantiating everything normally and we
# don't need other __magic__ methods from dataclass
@dataclasses.dataclass(init=True, repr=True, eq=True)
class Injection(PolymorphicMixin, CustomBase):
    EXPERIMENT_RUN_CLASS_ID_LAMBDA: typing.ClassVar[
        typing.Callable[..., sqlalchemy.Column[sqlalchemy.Integer]]
    ] = lambda: ExperimentRun.id_
    EXPERIMENT_RUN_CLASS_LAMBDA: typing.ClassVar[
        typing.Callable[..., typing.Type["CustomBaseClass"]]
    ] = lambda: ExperimentRun
    EXPERIMENT_RUN_BACKPOPULATES_NAME: typing.ClassVar[str] = "injections"

    # NOTE: cascading does not work if it's not a mixin or an abstract class
    # @sqlalchemy.orm.declared_attr.cascading
    @sqlalchemy.orm.declared_attr
    def experiment_run_id(cls) -> sqlalchemy.orm.Mapped[typing.Optional[int]]:
        return sqlalchemy.Column(
            sqlalchemy.ForeignKey(cls.EXPERIMENT_RUN_CLASS_ID_LAMBDA())
        )

    @sqlalchemy.orm.declared_attr
    def experiment_run(cls) -> sqlalchemy.orm.Mapped[typing.Optional["ExperimentRun"]]:
        return sqlalchemy.orm.relationship(
            cls.EXPERIMENT_RUN_CLASS_LAMBDA,
            back_populates=cls.EXPERIMENT_RUN_BACKPOPULATES_NAME,
        )

    @sqlalchemy.orm.declared_attr
    def location(self) -> sqlalchemy.orm.Mapped[typing.Any]:
        return sqlalchemy.Column(sqlalchemy.PickleType, nullable=False)

    @sqlalchemy.orm.declared_attr
    def internal_id(self) -> sqlalchemy.orm.Mapped[int]:
        return sqlalchemy.Column(sqlalchemy.Integer, nullable=False)


@sqlalchemy.orm.declarative_mixin
class FaultBaseMixin(object):
    pass


@dataclasses.dataclass(init=True, repr=True, eq=True)
class Fault(FaultBaseMixin, Injection):
    # not needed
    # PARENT_CLASS: typing.Type[CustomBaseClass] = Injection
    # ID_NAME: str = "fault_id"
    pass


@sqlalchemy.orm.declarative_mixin
class MonitorBaseMixin(object):
    @sqlalchemy.orm.declared_attr
    def payload(
        self,
    ) -> sqlalchemy.orm.Mapped[typing.Optional[typing.Dict[str, typing.Any]]]:
        return sqlalchemy.Column(
            sqlalchemy.ext.mutable.MutableDict.as_mutable(sqlalchemy.PickleType)
        )


@dataclasses.dataclass(init=True, repr=True, eq=True)
class Monitor(MonitorBaseMixin, Injection):
    pass


def set_sqlite_pragma(dbapi_connection, connection_record) -> None:
    # enable foreign keys
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def pysqlite_begin_emission_fix_on_connect(dbapi_connection, connection_record) -> None:
    # disable pysqlite's emitting of the BEGIN statement entirely.
    # also stops it from emitting COMMIT before any DDL.
    dbapi_connection.isolation_level = None


def sqlalchemy_begin_emission_pysqlite(conn) -> None:
    # emit our own BEGIN
    conn.exec_driver_sql("BEGIN")


# we call all the previous functions to connect all the event listeners from the engine
# if the listener already exists, we skip it
def fix_pysqlite(engine) -> None:
    if not sqlalchemy.event.contains(engine, "connect", set_sqlite_pragma):
        sqlalchemy.event.listen(engine, "connect", set_sqlite_pragma)

    if not sqlalchemy.event.contains(
        engine, "connect", pysqlite_begin_emission_fix_on_connect
    ):
        sqlalchemy.event.listen(
            engine, "connect", pysqlite_begin_emission_fix_on_connect
        )

    if not sqlalchemy.event.contains(
        engine, "begin", sqlalchemy_begin_emission_pysqlite
    ):
        sqlalchemy.event.listen(engine, "begin", sqlalchemy_begin_emission_pysqlite)
