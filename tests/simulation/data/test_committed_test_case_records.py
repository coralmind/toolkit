"""The committed test case records load against the real library."""

from __future__ import annotations

import pytest
from coralmap import KnowledgeLibrary, load

from corallab import paths
from corallab.simulation.data.test_case_records import (
    TestCaseRecordsError,
    check_test_case_records,
    load_test_case_records,
)

_REAL_KL = paths.default_repo_root().parent / "knowledge-library"
_RECORD_FILES = [
    "test_case_records/l3_launch_entrenchment.jsonl",
    "test_case_records/l3_launch_entrenchment_small.jsonl",
    "test_case_records/l3_launch_entrenchment_sample.jsonl",
    "test_case_records/examples/l1_launch_smoke.jsonl",
    "test_case_records/examples/l2_launch_smoke.jsonl",
    "test_case_records/examples/l3_launch_smoke.jsonl",
]


@pytest.fixture(scope="module")
def library() -> KnowledgeLibrary:
    if not (_REAL_KL / "entries").is_dir():
        pytest.skip(f"sibling Knowledge Library not found at {_REAL_KL}")
    return load(_REAL_KL)


@pytest.mark.parametrize("records_path", _RECORD_FILES)
def test_records_load_and_check_clean(records_path: str, library: KnowledgeLibrary) -> None:
    cases = load_test_case_records(records_path, "persona_records", library)
    assert cases
    assert check_test_case_records(cases, library) == []


@pytest.mark.parametrize("records_path", _RECORD_FILES)
def test_records_do_not_load_against_the_fixture(records_path: str) -> None:
    from corallab.kl import load_library

    with pytest.raises(TestCaseRecordsError, match="pins library"):
        load_test_case_records(records_path, "persona_records", load_library())


def test_record_files_exist() -> None:
    for records_path in _RECORD_FILES:
        assert (paths.data_dir() / records_path).is_file(), records_path
