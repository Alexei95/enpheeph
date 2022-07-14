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

import nox


nox.needs_version = ">=2022.1.7"
nox.options.default_venv_backend = "conda"
nox.options.reuse_existing_virtualenvs = False
nox.options.stop_on_first_error = False
nox.options.error_on_missing_interpreters = True
nox.options.sessions = ["test"]


# we do not need any virtualenv for this env wrapper
@nox.session(
    python=False,
    venv_backend=None,
)
def test(session):
    session.notify("_test_clean_coverage")
    session.notify("_test_pytest")
    session.notify("_test_report_coverage")


@nox.session(
    python=["3.9"],
    venv_backend="conda",
)
def _test_clean_coverage(session):
    session.install("coverage[toml]")
    session.run("python", "-m", "coverage", "erase")


@nox.session(
    python=["3.8", "3.9", "3.10"],
    venv_backend="conda",
)
def _test_pytest(session):
    # session.conda_install(
    #     "--only-deps", ".[dev]", "-c", "pytorch", "-c", "conda-forge"
    # )
    session.install("-e", ".[full-dev-cpu]")
    session.run(
        "python",
        "-m",
        "pytest",
        f"--junitxml={str(session.cache_dir)}/tools/pytest/junit-{session.name}.xml",
    )


@nox.session(
    python=["3.9"],
    venv_backend="conda",
)
def _test_report_coverage(session):
    session.install("coverage[toml]")
    session.run("python", "-m", "coverage", "report")
    session.run(
        "python",
        "-m",
        "coverage",
        "xml",
        "-o",
        f"{str(session.cache_dir)}/tools/coverage/coverage.xml",
    )
