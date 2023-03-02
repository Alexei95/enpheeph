# -*- coding: utf-8 -*-
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

import copy
import os
import pathlib

import nox

CI_RUN = "CI" in os.environ

nox.needs_version = ">=2022.11.21"
nox.options.envdir = ".nox"
nox.options.default_venv_backend = "virtualenv"
# nox.options.default_venv_backend = "mamba"
nox.options.error_on_external_run = True
nox.options.error_on_missing_interpreters = True
nox.options.reuse_existing_virtualenvs = False
# nox.options.sessions = ["test"]
nox.options.stop_on_first_error = True


def _select_cache_dir(session):
    if not session.interactive or CI_RUN:
        cache_dir = pathlib.Path(".logs")
    else:
        cache_dir = session.cache_dir
    return cache_dir


@nox.session
@nox.parametrize(
    "python", ["3.10"],
)
def test(session):
    cache_dir = _select_cache_dir(session)

    session.install("-e", ".[full-dev-cpu]")

    # clean previous coverage
    # if run here it would clear the other ones
    # session.run("python", "-m", "coverage", "erase")

    # run pytest with Jenkins and coverage output
    session.run(
        "python",
        "-m",
        "pytest",
        "--cov=enpheeph",
        "--cov-config=pyproject.toml",
        f"--junitxml={str(cache_dir)}/tools/pytest/junit-{session.name}-{session.python}.xml",
        *session.posargs,
        env={"COVERAGE_FILE": f".coverage.{session.name}"},
    )
    session.notify("coverage")

@nox.session
@nox.parametrize(
    "python", ["3.10"],
)
def coverage(session):
    cache_dir = _select_cache_dir(session)

    session.install("-e", ".[coverage]")

    # combine and report the coverage, to generate output file
    session.run("python", "-m", "coverage", "combine")
    session.run("python", "-m", "coverage", "report")
    session.run(
        "python",
        "-m",
        "coverage",
        "xml",
        "-o",
        f"{str(cache_dir)}/tools/coverage/coverage.xml",
    )

    # clean the coverage afterwards
    session.run("python", "-m", "coverage", "erase")
