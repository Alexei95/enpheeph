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


# we do not need any virtualenv for this env wrapper
@nox.session(
    python=False,
    venv_backend=None,
)
def test(session):
    session.notify("_test_clean_coverage_before")
    session.notify("_test_pytest")
    session.notify("_test_report_coverage")


@nox.session
@nox.parametrize(
    "python", ["3.10"],
)
def _test_clean_coverage_before(session):
    session.install("coverage[toml]")
    session.run("python", "-m", "coverage", "erase")


@nox.session
@nox.parametrize(
    "python", ["3.10"],
)
def _test_pytest(session):
    # session.conda_install(
    #     "--only-deps", ".[dev]", "-c", "pytorch", "-c", "conda-forge"
    # )
    if not session.interactive or CI_RUN:
        cache_dir = pathlib.Path(".logs")
    else:
        cache_dir = session.cache_dir
    session.install("-e", ".[full-dev-cpu]")
    session.run(
        "python",
        "-m",
        "pytest",
        "--cov=enpheeph",
        "--cov-config=pyproject.toml",
        f"--junitxml={str(cache_dir)}/tools/pytest/junit-{session.name}-{session.python}.xml",
        *session.posargs,
    )


@nox.session
@nox.parametrize(
    "python", ["3.10"],
)
def _test_report_coverage(session):
    if not session.interactive or CI_RUN:
        cache_dir = pathlib.Path(".logs")
    else:
        cache_dir = session.cache_dir
    session.install("coverage[toml]")
    session.run("python", "-m", "coverage", "report")
    session.run(
        "python",
        "-m",
        "coverage",
        "xml",
        "-o",
        f"{str(cache_dir)}/tools/coverage/coverage.xml",
    )
