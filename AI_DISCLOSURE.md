# AI disclosure

**Kirthan S (DA24B009)**, DA3408 AIOps Lab, course project "Guess the Elo".
*Written as required by the course "Fair and Responsible Use of AI" code of conduct, §4.*

## 1. Tools I used

| Tool | Role in this project |
|---|---|
| **Claude Code** (Anthropic; Claude Opus models) | My design-discussion partner, planner and reviewer. It also helped organize and write the code, documentation. |

## 2. How I used them

**Design.** I set the requirements and made the product and scope decisions. For each technical decision, Claude proposed options with alternatives and trade-offs, written as decision records linked to the course lectures (D1–D30). I accepted, changed or rejected each one.

The trade-offs I weighed myself, and where I changed or overruled Claude's proposal:

- **Frontend (U1).** I chose static HTML/JS with nginx over Streamlit and React. I wanted full control of the board replay, and the cleanest separation from the backend, which the rubric's loose-coupling requirement asks for.
- **Sample sizes (D21).** 20k games per training month and 10k per replay month. That was enough for stable monthly metrics at a one-off Stockfish cost I could run overnight.
- **Engine budget (D28).** The measured Stockfish speed meant about 14 hours at 20k nodes. I chose 5k nodes with the full sample, over 20k nodes with half the sample. Amateur skill differences are much larger than the gap between 5k-node and 20k-node evaluations, and the run then fitted overnight with room for one re-run.
- **Drift thresholds (D30).** The original thresholds (MAE > 1.15×, PSI > 0.25) would never fire on real 2020 data. I chose thresholds calibrated from measured noise (1.04× MAE, about 4–5 standard errors; PSI 0.05), plus a drift-injection switch for demos that is **disclosed and off by default**. The final replay did not use the switch: the alert fired on real drift in April 2020.

**Debugging.** Failures from real runs were diagnosed from the actual logs by Codex and Claude. They included Docker networking, DVC run-cache behaviour, a UID-lookup crash in the trainer container, Spark and Ray resource limits, and an MLflow container that stopped overnight. Each one is recorded with its root cause in the project journal.


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
