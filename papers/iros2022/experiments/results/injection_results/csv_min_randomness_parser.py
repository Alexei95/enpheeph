# -*- coding: utf-8 -*-
# enpheeph - Neural Fault Injection Framework
# Copyright (C) 2020-2023 Alessio "Alexei95" Colucci
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

import pathlib
import sys

import pandas

csv_path = pathlib.Path(sys.argv[1])
dest_folder = pathlib.Path(sys.argv[2])
dest_folder.mkdir(parents=True, exist_ok=True)

csv = pandas.read_csv(csv_path)
split_csv = {}
for r in csv["randomness"].unique().tolist():
    split_csv[r] = csv[csv["randomness"] == r]

randomness_csv = {}
for r, c in split_csv.items():
    randomness_csv[r] = c.iloc[[c["test_accuracy"].argmin()]]

randomness = pandas.concat(randomness_csv.values())

randomness.to_csv(dest_folder / "randomness.csv")
