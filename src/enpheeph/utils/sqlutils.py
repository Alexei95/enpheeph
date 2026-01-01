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
