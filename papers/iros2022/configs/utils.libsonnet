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

{
    # join paths without / repetition
    # for PyTorch/PyTorch Lightning is not strictly required
    joinPath(a, b)::
        if std.endsWith(a, "/") then
            a + b
        else
            a + "/" + b
    ,

    safeGet(o, f, default=null, include_hidden=true)::
        if include_hidden then
            if std.objectHasAll(o, f) then
                o[f]
            else
                default
        else
            if std.objectHas(o, f) then
                o[f]
            else
                default
    ,
}
