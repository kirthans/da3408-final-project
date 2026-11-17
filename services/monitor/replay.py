"""Local replay controls; the Make targets load API_KEY from .env."""

import json
import os
import sys

import requests

action = sys.argv[1]
monitor = "http://127.0.0.1:8002"
if action in {"play", "pause"}:
    response = requests.post(
        monitor + "/clock/" + action,
        headers={"Authorization": "Bearer " + os.environ["API_KEY"]},
        timeout=30,
    )
    response.raise_for_status()
for service, base, fields in [
    (
        "monitor",
        monitor,
        (
            "current_month",
            "revealed",
            "playing",
            "seconds_to_next_reveal",
            "champion_version",
            "previous_champion_version",
        ),
    ),
    ("trainer", "http://127.0.0.1:8001", ("in_progress", "current_run", "champion_version")),
]:
    response = requests.get(base + "/status", timeout=30)
    response.raise_for_status()
    status = response.json()
    print(
        service + ": " + json.dumps({key: status.get(key) for key in fields}, separators=(",", ":"))
    )
