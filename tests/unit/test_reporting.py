"""Evidence joins and JUnit parsing must not invent successful measurements."""

import importlib.util
from pathlib import Path

import pytest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


report = module("test_report")
ac5 = module("ac5_evidence")


def test_junit_outcomes(tmp_path):
    xml = tmp_path / "tests.xml"
    xml.write_text("""<testsuites><testsuite><testcase classname="tests.unit.test_a"/>
    <testcase classname="tests.unit.test_a"><failure/></testcase>
    <testcase file="tests/contract/test_b.py"><skipped/></testcase>
    <testcase file="tests/contract/test_b.py"><error/></testcase></testsuite></testsuites>""")
    parsed = report.parse_junit(xml)
    assert parsed["totals"] == dict(total=4, passed=1, failed=1, skipped=1, errors=1)
    assert parsed["files"]["tests/unit/test_a.py"]["failed"] == 1
    assert report.test_verdict(parsed["files"], "tests/contract/")[1] == "FAIL"
    assert "not available" in report.generate(tmp_path)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("/home/another-user/checkout/tests/unit/test_a.py", "tests/unit/test_a.py"),
        ("tests/unit/test_a.py", "tests/unit/test_a.py"),
        ("/private/checkout/test_a.py", "test_a.py"),
    ],
)
def test_junit_report_uses_relative_paths(tmp_path, filename, expected):
    reports = tmp_path / "reports"
    reports.mkdir()
    xml = reports / "tests.xml"
    xml.write_text(f'<testsuite><testcase file="{filename}"/></testsuite>')
    parsed = report.parse_junit(xml)
    assert list(parsed["files"]) == [expected]
    generated = report.generate(tmp_path)
    assert f"| {expected} |" in generated
    assert "/home/another-user" not in generated
    assert "/private/checkout" not in generated


def synthetic():
    return {
        "month": [{"values": [[10, "202004"], [20, "202005"], [30, "202005"]]}],
        "version": [{"values": [[10, "4"], [20, "5"], [30, "5"]]}],
        "mae": [
            {"metric": {"model": "champion"}, "values": [[10, "200"], [20, "192"], [30, "190"]]},
            {"metric": {"model": "previous_champion"}, "values": [[20, "198"], [30, "197"]]},
        ],
    }


def promotion():
    return dict(
        run_id="run",
        trigger="drift_alert",
        status="promoted",
        registry_version="5",
        replay_month="2020-04",
        ended_at=15,
    )


def test_last_month_sample_and_promotion():
    evidence = ac5.reconstruct(synthetic(), [promotion()])
    assert evidence["verdict"] == "PASS"
    assert evidence["months"][1]["champion"] == 190
    assert evidence["promotions"][0]["previous_champion_mae"] == 197
    assert ac5.reconstruct({}, [promotion()])["verdict"] == "pending"
    assert (
        ac5.reconstruct(synthetic(), [dict(promotion(), trigger="manual")])["verdict"] == "pending"
    )


@pytest.mark.parametrize("new,version,expected", [(198, "5", "FAIL"), (190, "6", "pending")])
def test_ac5_failure_or_superseded_version(new, version, expected):
    series = synthetic()
    series["mae"][0]["values"][-1][1] = str(new)
    series["version"][0]["values"][-1][1] = version
    assert ac5.reconstruct(series, [promotion()])["verdict"] == expected


def test_replay_reset_excludes_earlier_generation():
    reset = dict(status="promoted", reset=True, ended_at=25, replay_generation=0)
    evidence = ac5.reconstruct(synthetic(), [promotion(), reset])
    assert evidence["promotions"] == []
    assert len(evidence["months"]) == 1
    assert evidence["months"][0]["sampled_at"] == 30
    assert evidence["verdict"] == "pending"
