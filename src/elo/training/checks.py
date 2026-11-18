"""Small test-summary and acceptance-gate commands used by DVC and Make."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from elo.training.dataset import write_json


def sanitize_junit(path, repository_root=None):
    """Remove machine-specific hostnames and repository paths from pytest XML."""
    repository_root = Path(repository_root or Path(__file__).resolve().parents[3]).resolve()
    prefix = str(repository_root)

    def relative(value):
        return value.replace(prefix + "/", "").replace(prefix, ".")

    tree = ET.parse(path)
    for element in tree.iter():
        for key, value in element.attrib.items():
            element.set(key, "localhost" if key == "hostname" else relative(value))
        if element.text:
            element.text = relative(element.text)
        if element.tail:
            element.tail = relative(element.tail)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return tree.getroot()


if __name__ == "__main__":
    if sys.argv[1] == "tests":
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", "not engine", "--junitxml=reports/tests.xml"]
        )
        root = sanitize_junit("reports/tests.xml")
        cases = root.findall(".//testcase")
        failed = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
        skipped = sum(c.find("skipped") is not None for c in cases)
        write_json(
            "reports/tests.json",
            dict(
                total=len(cases),
                passed=len(cases) - failed - skipped,
                failed=failed,
                skipped=skipped,
            ),
        )
        sys.exit(result.returncode)
    metrics = json.loads(Path("reports/metrics.json").read_text())
    sys.exit(0 if metrics["ac1_pass"] and metrics["ac2_pass"] else 1)
