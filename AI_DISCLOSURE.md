# AI disclosure

**Kirthan S (DA24B009)**, DA3408 AIOps Lab, course project "Guess the Elo".
*Written as required by the course "Fair and Responsible Use of AI" code of conduct, §4.*

## 1. Tools I used

| Tool | Role in this project |
|---|---|
| **Claude Code** (Anthropic; Claude Opus models) | My design-discussion partner, planner and reviewer. It also wrote the documentation: `DESIGN.md`, `docs/*.md`, the report and the project journal. |
| **Codex** (OpenAI CLI, model `gpt-6.1-sol`), which Claude drove through the `agent-link` MCP bridge | Wrote the implementation: Python packages, services, Dockerfiles, Compose, DVC stages, the Prometheus, Alertmanager and Grafana configuration, the frontend, and the tests. |

I set the process up deliberately:
- Claude was not allowed to write implementation code. It planned, delegated to Codex and reviewed Codex's work.
- Neither tool could commit, push or deploy without my explicit permission.
- I reviewed the system at four gates before any work moved on.

## 2. How I used them

**Design (2026-10-26 to 2026-11-15).** I set the requirements and made the product and scope decisions. For each technical decision, Claude proposed options with alternatives and trade-offs, written as decision records linked to the course lectures (D1–D30). I accepted, changed or rejected each one.

The trade-offs I weighed myself, and where I changed or overruled Claude's proposal:

- **Frontend (U1).** I chose static HTML/JS with nginx over Streamlit and React. I wanted full control of the board replay, and the cleanest separation from the backend, which the rubric's loose-coupling requirement asks for.
- **Data window (U3).** I chose to train on late 2019 and replay 2020 as "production". The lockdown and *Queen's Gambit* influx is real, well-documented drift, so a drift alert had a genuine chance of firing on its own, without faking it.
- **Kubernetes (U4).** I asked Claude to explain Kubernetes first, then kept it out of scope. Everything runs on one laptop, Compose is what the rubric grades, and Spark and Ray already cover data engineering. I wanted a Compose→K8s mapping in the HLD so I could answer "how would this scale?".
- **Model ambition (U2 → D24).** At first I wanted a GRU challenger next to XGBoost. When the deadline got tight, Claude proposed three cuts: drop the GRU, shrink the data, or merge the monitor into the trainer. **I accepted only dropping the GRU and rejected the other two.** I wanted the full data and a separate monitor service, because they matter more to the AIOps story than a second model family.
- **Scope (D26).** I asked whether a codebase of about 7,000 lines was overkill. We cut it to a lean version of the same architecture (about 3.5–4k lines): no feedback endpoint, PSI without KS, two dashboards instead of three. The rubric rewards each component working, and I have to be able to defend every line in the viva.
- **Sample sizes (D21).** 20k games per training month and 10k per replay month. That was enough for stable monthly metrics at a one-off Stockfish cost I could run overnight.
- **Storage (D27).** MinIO's Docker images disappeared during the build. Of the replacements, I chose an **SSH DVC remote** over the alternatives (a frozen MinIO image, SeaweedFS, RustFS). SSH is taught in the course, needs no unmaintained container, and keeps credentials out of the stack. I set up `sshd` and the restricted key on my machine myself.
- **Engine budget (D28).** The measured Stockfish speed meant about 14 hours at 20k nodes. I chose 5k nodes with the full sample, over 20k nodes with half the sample. Amateur skill differences are much larger than the gap between 5k-node and 20k-node evaluations, and the run then fitted overnight with room for one re-run.
- **Drift thresholds (D30).** The original thresholds (MAE > 1.15×, PSI > 0.25) would never fire on real 2020 data. I chose thresholds calibrated from measured noise (1.04× MAE, about 4–5 standard errors; PSI 0.05), plus a drift-injection switch for demos that is **disclosed and off by default**. The final replay did not use the switch: the alert fired on real drift in April 2020.
- **Trainer commits (D22).** I decided the trainer must never commit to git by itself. It pushes data with DVC and marks the run "pending commit", and I commit the new `params.yaml`/`dvc.lock`.
- **Gate D changes.** I approved:
  - turning on MLflow autolog in the sweep, alongside the explicit logging;
  - excluding opening moves from the "moves that shaped the estimate" list. An example like "1. d4: weaker play" convinced me they mostly reflect memorised theory, not skill.

**Implementation (2026-11-15 to 2026-11-18).**
- The approved design was split into milestones. For each one, Claude wrote a delegation brief with scope, constraints and acceptance criteria, and Codex implemented it and ran the tests.
- Claude reviewed every milestone by reading the code and re-running the checks itself, and sent fixes back to Codex.
- I reviewed the result at each gate: A (data pipeline), B (training and serving), C (monitoring and retraining), D (console, tests, documentation). Each gate had a written report with a code walkthrough.
- **No metric, log or dashboard in this repository was written by hand or invented by an AI.** Every number comes from running the real pipeline, services and tests on Lichess data: `reports/*.json`, `reports/tests.xml`, MLflow runs and Prometheus.
- **AI-generated PromQL:** every alert expression in `infra/prometheus/rules.yml` has a comment with the prompt that produced it, as the course requires.

**Debugging.** Failures from real runs were diagnosed from the actual logs by Codex and Claude. They included Docker networking, DVC run-cache behaviour, a UID-lookup crash in the trainer container, Spark and Ray resource limits, and an MLflow container that stopped overnight. Each one is recorded with its root cause in the project journal.

**Documentation.**
- Claude drafted `DESIGN.md`, `docs/architecture.md`, `hld.md`, `lld.md`, `test_plan.md`, `user_manual.md` and the report (`docs/report.md`, built into `docs/report.pdf`). It also produced the report's data-profile chart and the MLflow screenshots from the real report files and the live MLflow server.
- `docs/test_report.md` is generated by a script from the real test output.
- Codex wrote the docstrings and inline comments in a documentation-only pass. Claude checked that the code itself did not change: every Python file's syntax tree is identical apart from docstrings.
- **What I did myself:** I reviewed the report and the documents, and I recorded and narrated the demo video.

## 3. Review and accountability

- I reviewed the system at four gates (A–D), each with a report, a code walkthrough and a checklist, and I approved each one before work continued. I also approved every deviation from the design (`DESIGN.md` §19–§24).
- I checked the security-relevant settings explicitly:
  - secrets only in the gitignored `.env`;
  - an API key on the retrain endpoints;
  - monitoring and admin ports bound to localhost;
  - an upload size cap;
  - TLS on nginx;
  - no personal paths in committed files.
- **I have reviewed the code and configuration and can explain, modify and troubleshoot every part of it, including the orchestration pipeline, data versioning and model tracking.**

## 4. Impact

- AI tools wrote nearly all of the code and documentation text. That made a system of this scope possible within the course deadline: ten services, Spark and Ray, DVC, MLflow, monitoring, and a drift-triggered retraining loop.
- The architecture, the scope cuts and the evaluation criteria came from design discussions in which I made the decisions. These are recorded with dates in the decision log.
- In my own words: **the AI did the typing, but the trade-offs, the scope and the responsibility for the result are mine, and working with it this way taught me more about how the MLOps pieces fit together than I would have learned from building a smaller system alone.**

## 5. Where the records are

I keep the design and process records myself, and they are available to the instructors on request. They are not part of this repository:
- `DECISIONS.md`: every design decision, with options, rationale and who decided.
- `JOURNAL.md`: a timestamped log of every discussion, delegation, review and result.
- The delegation briefs: the exact tasks given to Codex, and its reports.

The repository contains committed evidence of them: `DESIGN.md` (the approved design, with its decision IDs and change sections §19–§24) and the prompt comments on the alert rules in `infra/prometheus/rules.yml`.
