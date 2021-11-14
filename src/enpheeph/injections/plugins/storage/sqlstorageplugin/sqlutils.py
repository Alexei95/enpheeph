# -*- coding: utf-8 -*-
# import sqlalchemy
# import sqlalchemy.dialects.postgresql
# import sqlalchemy.ext.compiler
# import sqlalchemy.sql.functions
# import sqlalchemy.types


# to have utc timestamps in UTC for the database
# the default function func.time() returns the local time
# UTC TIMESTAMP SQL
# class utcnow(sqlalchemy.sql.functions.FunctionElement):
#     type = sqlalchemy.types.DateTime()


# @sqlalchemy.ext.compiler.compiles(utcnow, "postgresql")
# def pg_utcnow(element, compiler, **kw):
#     return "TIMEZONE('utc', CURRENT_TIMESTAMP)"


# END UTC TIMESTAMP SQL
