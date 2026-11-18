"""Committed JUnit evidence must not include host-specific checkout details."""

import xml.etree.ElementTree as ET

import pytest

from elo.training.checks import sanitize_junit


@pytest.mark.parametrize("wrapper", ["testsuite", "testsuites"])
def test_sanitize_junit(tmp_path, wrapper):
    repository = tmp_path / "home" / "developer" / "repo & checkout"
    location = str(repository / "tests/integration/smoke_test.py") + ":16"
    root = ET.Element(wrapper, hostname="workstation", tests="2", failures="0", skipped="1")
    suite = ET.SubElement(root, "testsuite", hostname="worker") if wrapper == "testsuites" else root
    case = ET.SubElement(suite, "testcase", file=location, name="smoke", time="0.001")
    skipped = ET.SubElement(case, "skipped", message=f"{location}: requires service")
    skipped.text = f"Skipped at {location} (root {repository})"
    skipped.tail = f"Output from {location}"
    ET.SubElement(suite, "testcase", name="passed")
    path = tmp_path / "tests.xml"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)

    sanitized = sanitize_junit(path, repository)

    assert str(repository) not in path.read_text()
    assert "workstation" not in path.read_text()
    assert "worker" not in path.read_text()
    assert sanitized.attrib == dict(hostname="localhost", tests="2", failures="0", skipped="1")
    assert all(s.get("hostname") == "localhost" for s in sanitized.iter("testsuite"))
    case, passed = sanitized.iter("testcase")
    assert case.attrib == dict(
        file="tests/integration/smoke_test.py:16", name="smoke", time="0.001"
    )
    assert passed.get("name") == "passed"
    skipped = case.find("skipped")
    assert skipped.get("message") == "tests/integration/smoke_test.py:16: requires service"
    assert skipped.text == "Skipped at tests/integration/smoke_test.py:16 (root .)"
    assert skipped.tail == "Output from tests/integration/smoke_test.py:16"
    before = path.read_bytes()
    sanitize_junit(path, repository)
    assert path.read_bytes() == before
