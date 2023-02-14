#!/usr/bin/env python
# coding: utf8

""" Unit testing for Separator class. """

__email__ = "research@deezer.com"
__author__ = "Deezer Research"
__license__ = "MIT License"

from typer.testing import CliRunner

from spleeter.__main__ import spleeter


def test_version() -> None:

    runner = CliRunner()

    # execute spleeter version command
    result = runner.invoke(
        spleeter,
        [
            "--version",
        ],
    )
