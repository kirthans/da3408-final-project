"""Wait for the authenticated replay reset so the Make target has completion semantics."""

import os
import time

import requests

base = os.getenv("TRAINER_ADMIN_URL", "http://localhost:8001")
response = requests.post(
    base + "/replay/reset", headers={"Authorization": "Bearer " + os.environ["API_KEY"]}, timeout=30
)
response.raise_for_status()
run_id = response.json()["run_id"]
while True:
    response = requests.get(base + "/runs/" + run_id, timeout=30)
    response.raise_for_status()
    run = response.json()
    if run["status"] != "running":
        print(run)
        if run["status"] != "promoted":
            raise SystemExit(1)
        break
    time.sleep(2)
