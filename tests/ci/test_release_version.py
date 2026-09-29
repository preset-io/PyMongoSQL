"""The published version must equal the one the build backend writes."""

import runpy
from pathlib import Path

import pytest

module = runpy.run_path(str(Path(__file__).resolve().parents[2] / "ci" / "release_version.py"))
release_version = module["release_version"]
declared_version = module["declared_version"]


@pytest.mark.parametrize(
    "change,revision,expected",
    [
        (None, None, "0.7.4.1"),
        ("3", "e2bc688f1a2b", "0.7.4.1+pr.3.e2bc688f1a2b"),
        ("3", "ABCDEF123456", "0.7.4.1+pr.3.abcdef123456"),
        ("3", "012345678901", "0.7.4.1+pr.3.12345678901"),
    ],
)
def test_release_version_is_normalized(change, revision, expected):
    assert release_version("0.7.4.1", change, revision) == expected


@pytest.mark.parametrize("change,revision", [("PR-3", "e2bc688"), ("3", "not-a-sha"), ("3", None)])
def test_bad_pull_request_inputs_fail(change, revision):
    with pytest.raises(SystemExit):
        release_version("0.7.4.1", change, revision)


def test_declared_version_is_a_four_part_release():
    assert declared_version('__version__: str = "0.7.4.1"\n') == "0.7.4.1"
    assert len(declared_version().split(".")) == 4


@pytest.mark.parametrize("source", ['__version__: str = "0.7.4"\n', '__version__: str = "0.7.4.1+x"\n', ""])
def test_declared_version_rejects_other_forms(source):
    with pytest.raises(SystemExit):
        declared_version(source)
