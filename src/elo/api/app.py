"""Thin FastAPI boundary around shared PGN/features, Stockfish and MLflow REST."""

import hashlib
import json
import os
import queue
import time
import uuid
from contextlib import asynccontextmanager
from email.parser import BytesParser
from email.policy import default
from typing import Literal

import chess
import pandas as pd
import requests
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException

from elo.api import metrics
from elo.common.config import load_params
from elo.common.errors import (
    EloError,
    EngineError,
    InvalidPgnError,
    ModelUnavailableError,
    PayloadTooLargeError,
)
from elo.common.logging import get_logger, request_id
from elo.engine import EngineAnalyzer
from elo.features.moves import build_move_rows
from elo.pgn import PgnParser, sanitize_headers

LOG = get_logger(__name__)
NAME = "elo-estimator"


class ApiError(EloError):
    """Carry a public error code and HTTP status through the domain handler (DESIGN §9)."""

    def __init__(self, code, message, status=422):
        """Initialize the message, stable code and HTTP status.

        Args:
            code: Machine-readable public error code.
            message: User-facing explanation.
            status: HTTP response status, defaulting to validation failure."""
        super().__init__(message)
        self.code, self.http_status = code, status


class PgnBody(BaseModel):
    """Validate a strictly typed PGN request body (DESIGN §9)."""

    model_config = ConfigDict(strict=True)
    pgn: str


class PredictBody(PgnBody):
    """Validate game selection, influence count and optional time-control fallback."""

    game_index: int = Field(default=0, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)
    tc_class: Literal["blitz", "rapid", "classical"] | None = None


def tc_class(base, inc=0):
    """Mirror M2's exact base + 40*increment boundaries."""
    if base == "-":
        return "correspondence"
    estimate = base + 40 * inc
    for bound, label in [(29, "ultrabullet"), (179, "bullet"), (479, "blitz"), (1499, "rapid")]:
        if estimate < bound:
            return label
    return "classical"


def time_control(header):
    """Decode a PGN time control into base seconds, increment and class.

    Args:
        header: TimeControl value; missing or unknown values have no class.

    Returns:
        A tuple of base seconds, increment seconds and time-control class.

    Raises:
        ApiError: The header has malformed or negative numeric fields."""
    if not header or header == "?":
        return None, 0, None
    if header == "-":
        return None, 0, "correspondence"
    try:
        parts = header.split("+")
        base, inc = int(parts[0]), int(parts[1]) if len(parts) == 2 else 0
        if len(parts) > 2 or base < 0 or inc < 0:
            raise ValueError
        return base, inc, tc_class(base, inc)
    except ValueError as exc:
        raise ApiError("unsupported_time_control", "Unrecognized TimeControl header") from exc


def game_entry(game, index):
    """Build display metadata for a parsed game without passing identities to the model."""
    h, moves = game["headers"], game["moves"]
    return dict(
        index=index,
        white=h.get("White"),
        black=h.get("Black"),
        date=h.get("Date"),
        event=h.get("Event"),
        time_control=h.get("TimeControl"),
        tc_class=time_control(h.get("TimeControl"))[2],
        plies=len(moves),
        has_clock=any(m["clock_s"] is not None for m in moves),
        result=h.get("Result"),
    )


def remote(url, *, payload=None, retry=False, code="model_unavailable", headers=None):
    """Call an upstream with finite timeouts and optional retries (DESIGN §9).

    Args:
        url: Full upstream endpoint.
        payload: JSON body for POST; None selects GET.
        retry: Allow up to three attempts for network errors and 5xx only.
        code: Public error code for upstream unavailability.
        headers: Optional authentication headers to forward.

    Returns:
        The successful requests response.

    Raises:
        ApiError: The upstream rejects the call or exhausts attempts. Trainer
            401 and 409 retain their public statuses (DESIGN §24 F1)."""
    model_call = url.startswith(os.getenv("MODEL_SERVER_URL", "http://localhost:5001") + "/")
    for attempt in range(3 if retry else 1):
        try:
            response = requests.request(
                "POST" if payload is not None else "GET",
                url,
                json=payload,
                timeout=(3, 120),
                headers=headers,
            )
            if response.status_code < 500:
                if not response.ok:
                    # Preserve actionable auth/conflict errors rather than hiding them as 503 (F1).
                    if code == "trainer_unavailable" and response.status_code in {401, 409}:
                        raise ApiError(
                            "unauthorized" if response.status_code == 401 else "run_in_progress",
                            (
                                "Invalid API key"
                                if response.status_code == 401
                                else "A run is already in progress"
                            ),
                            response.status_code,
                        )
                    if model_call:
                        metrics.MODEL_ERRORS.inc()
                    raise ApiError(code, "Upstream rejected the request", 503)
                return response
        except requests.RequestException:
            pass
        # Retry network failures/5xx only, with 0.1s then 0.2s backoff (DESIGN §9).
        if retry and attempt < 2:
            time.sleep(0.1 * 2**attempt)
    if model_call:
        metrics.MODEL_ERRORS.inc()
    raise ApiError(code, "Service is unavailable", 503)


def model_metadata():
    """Resolve champion metadata and metric tags through MLflow REST (DESIGN D11)."""
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000").rstrip("/")
    response = remote(f"{uri}/api/2.0/mlflow/registered-models/alias?name={NAME}&alias=champion")
    version = response.json()["model_version"]
    tags = {tag["key"]: tag["value"] for tag in version.get("tags", [])}
    return dict(
        name=NAME,
        version=str(version["version"]),
        run_id=version["run_id"],
        aliases=version.get("aliases", ["champion"]),
        metrics={k: float(tags[k]) if k in tags else None for k in ("test_mae", "coverage")},
        created=str(version.get("creation_timestamp", "")),
    )


@asynccontextmanager
async def lifespan(app):
    """Own the bounded serving engine pool and metrics exporter (DESIGN D16).

    Args:
        app: FastAPI application receiving the engine queue.

    Raises:
        RuntimeError: Serving and training node budgets would differ."""
    metrics.start_metrics_server()
    pool = queue.Queue()
    engines = []
    params = load_params()["engine"]
    if params["nodes"] != 5000:
        raise RuntimeError("Serving nodes must match training (5000)")
    try:
        for _ in range(int(os.getenv("ENGINE_POOL", "2"))):
            engine = EngineAnalyzer(**params)
            engines.append(engine)
            pool.put(engine)
        app.state.engines = pool
        yield
    finally:
        for engine in engines:
            engine.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:8080,https://localhost:8443"
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


def error_response(request, code, message, status):
    """Build the public error envelope with its correlation ID (DESIGN §9)."""
    if 400 <= status < 500:
        metrics.PARSE_ERRORS.labels(reason=code).inc()
    return JSONResponse(
        {"error": {"code": code, "message": message, "request_id": request.state.request_id}},
        status_code=status,
        headers={"X-Request-ID": request.state.request_id},
    )


@app.middleware("http")
async def request_context(request, call_next):
    """Bound streamed bodies, propagate request IDs and record request metrics (DESIGN D20)."""
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    token = request_id.set(rid)
    started = time.monotonic()
    metrics.INFLIGHT.inc()
    response = None
    try:
        # Bound streamed uploads too, before validation or multipart parsing allocates them.
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > int(os.getenv("MAX_PGN_BYTES", "1048576")):
                response = error_response(
                    request, "payload_too_large", "PGN exceeds upload limit", 413
                )
                return response
        request._body = bytes(body)
        request.state.body = bytes(body)
        try:
            response = await call_next(request)
        except Exception:
            LOG.exception("Unhandled request failure")
            response = error_response(request, "internal_error", "Internal server error", 500)
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        endpoint = getattr(request.scope.get("route"), "path", "other")
        metrics.REQUESTS.labels(endpoint, str(response.status_code if response else 500)).inc()
        metrics.LATENCY.labels(endpoint).observe(time.monotonic() - started)
        metrics.INFLIGHT.dec()
        LOG.info("%s %s seconds=%.3f", request.method, request.url.path, time.monotonic() - started)
        request_id.reset(token)


@app.exception_handler(EloError)
def domain_error(request, exc):
    """Translate domain codes and statuses into the public error envelope (DESIGN §9)."""
    return error_response(request, exc.code, str(exc), exc.http_status)


@app.exception_handler(RequestValidationError)
@app.exception_handler(ValidationError)
def validation_error(request, exc):
    """Hide validation internals behind a stable 422 response (DESIGN §9)."""
    return error_response(request, "validation_error", "Invalid request fields", 422)


@app.exception_handler(HTTPException)
def http_error(request, exc):
    """Map routing errors to distinct public codes and preserve headers (DESIGN §24 F2)."""
    code, message = {
        404: ("not_found", "No such endpoint"),
        405: ("method_not_allowed", "Method not allowed"),
    }.get(exc.status_code, ("validation_error", str(exc.detail)))
    response = error_response(request, code, message, exc.status_code)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


@app.exception_handler(Exception)
def internal_error(request, exc):
    """Log unexpected failures and return a generic 500 error with a request ID."""
    LOG.exception("Unhandled request failure")
    return error_response(request, "internal_error", "Internal server error", 500)


@app.get("/health")
def health():
    """Return process liveness without contacting dependencies."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    """Check that an engine is available and the model server responds (DESIGN D18).

    Args:
        request: Request carrying the application engine queue.

    Returns:
        Readiness flags for the model server and engine.

    Raises:
        EngineError: The engine queue is unavailable or empty.
        ApiError: The model server cannot be reached."""
    if not hasattr(request.app.state, "engines") or request.app.state.engines.empty():
        raise EngineError("No engine is available")
    remote(os.getenv("MODEL_SERVER_URL", "http://localhost:5001") + "/ping")
    return dict(status="ready", model_server=True, engine=True)


@app.post("/v1/games/parse")
def parse_games(request: Request):
    """Parse UTF-8 multipart or JSON PGN into selectable game summaries (DESIGN §9).

    Args:
        request: Request with the size-bounded body cached by middleware.

    Returns:
        A games list with display metadata and time controls.

    Raises:
        ApiError: The multipart file or time-control fields are invalid.
        InvalidPgnError: The upload is not UTF-8 or contains invalid chess moves."""
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
            + request.state.body
        )
        parts = [
            part
            for part in message.iter_parts()
            if part.get_param("name", header="content-disposition") == "file"
        ]
        if len(parts) != 1:
            raise ApiError("validation_error", "Expected multipart file")
        try:
            text = parts[0].get_payload(decode=True).decode("utf-8-sig")
        except UnicodeError as exc:
            raise InvalidPgnError("PGN must be UTF-8") from exc
    else:
        text = PgnBody.model_validate_json(request.state.body).pgn
    return {"games": [game_entry(g, i) for i, g in enumerate(PgnParser().parse(text))]}


@app.get("/v1/model")
def model():
    """Expose champion metadata, mapping malformed upstream data to model unavailability."""
    try:
        return model_metadata()
    except (KeyError, ValueError, requests.RequestException) as exc:
        raise ModelUnavailableError("Model metadata is unavailable") from exc


@app.post("/v1/predict")
def predict(body: PredictBody, request: Request):
    """Analyze a selected game and request ratings, intervals and move influence.

    The model server owns aggregation, conformal calibration and occlusion
    (DESIGN D9, D10); rating headers are used only for display after prediction.

    Args:
        body: Validated PGN, game index, top-k and fallback time-control class.
        request: Request carrying its correlation ID and pooled engines.

    Returns:
        Ratings for both sides, move details, influence and model provenance.

    Raises:
        PayloadTooLargeError: PGN bytes exceed the configured cap.
        ApiError: Selection, game length, time control or upstream calls fail.
        InvalidPgnError: The PGN cannot be parsed legally.
        EngineError: No engine is available or analysis fails.
        ModelUnavailableError: The model response is incomplete or malformed."""
    metrics.PGN_BYTES.observe(len(body.pgn.encode()))
    if len(body.pgn.encode()) > int(os.getenv("MAX_PGN_BYTES", "1048576")):
        raise PayloadTooLargeError("PGN exceeds upload limit")
    games = PgnParser().parse(body.pgn)
    if body.game_index >= len(games):
        raise ApiError("game_index_out_of_range", "Game index is out of range")
    game = games[body.game_index]
    entry = game_entry(game, body.game_index)
    headers, labels = sanitize_headers(game["headers"])
    base, inc, tc = time_control(headers.get("TimeControl"))
    warnings = []
    if tc is None:
        tc = body.tc_class or "classical"
        warnings.append(f"No TimeControl header; using {tc} time-control class.")
    if tc not in {"blitz", "rapid", "classical"}:
        raise ApiError("unsupported_time_control", "This time control is unsupported")
    entry["tc_class"] = tc
    if any(sum(m["side"] == s for m in game["moves"]) < 10 for s in ("w", "b")):
        raise ApiError("game_too_short", "At least 10 plies per side are required")
    meta = dict(
        game_id=hashlib.sha256(body.pgn.encode()).hexdigest()[:16],
        tc_base=base,
        tc_inc=inc,
        tc_class=tc,
        result=headers.get("Result"),
        termination=headers.get("Termination", ""),
        eco=headers.get("ECO", ""),
        n_plies=len(game["moves"]),
        has_clock=entry["has_clock"],
        fen=game["fen"],
    )
    try:
        engine = request.app.state.engines.get(timeout=5)
    except queue.Empty as exc:
        raise EngineError("Engine pool is busy") from exc
    started = time.monotonic()
    try:
        analysis = engine.analyse_game(chess.Board(game["fen"]), game["moves"])
    finally:
        metrics.ENGINE.observe(time.monotonic() - started)
        request.app.state.engines.put(engine)
    seconds = time.monotonic() - started
    rows = build_move_rows(meta, game["moves"], analysis)
    frame = pd.DataFrame([{**row, **{k: v for k, v in meta.items() if k != "fen"}} for row in rows])
    payload = {
        "dataframe_split": json.loads(frame.to_json(orient="split")),
        "params": {"top_k": body.top_k},
    }
    payload["dataframe_split"].pop("index", None)
    response = remote(
        os.getenv("MODEL_SERVER_URL", "http://localhost:5001") + "/invocations",
        payload=payload,
        retry=True,
    )
    try:
        predictions = response.json()["predictions"]
        sides = {p["side"]: {k: int(p[k]) for k in ("elo", "lo", "hi")} for p in predictions}
        influence = [m for p in predictions for m in json.loads(p["influence"])]
        influence = sorted(influence, key=lambda m: -abs(m["delta_elo"]))[: body.top_k]
        metadata = model()
        if set(sides) != {"w", "b"}:
            raise ValueError("Missing model predictions")
    except (KeyError, ValueError, TypeError) as exc:
        metrics.MODEL_ERRORS.inc()
        raise ModelUnavailableError("Invalid model response") from exc

    for side, result in sides.items():
        metrics.RATING.labels(side, tc).observe(result["elo"])
        metrics.INTERVAL.observe(result["hi"] - result["lo"])

    def rating(key):
        """Read a display-only PGN rating, returning None for absent or invalid labels."""
        try:
            return int((labels or {})[key])
        except (KeyError, ValueError):
            return None

    return dict(
        request_id=request.state.request_id,
        game=entry,
        white=sides["w"],
        black=sides["b"],
        interval_level=0.8,
        pgn_ratings=dict(white=rating("WhiteElo"), black=rating("BlackElo")),
        moves=[
            {
                **{k: row[k] for k in ("ply", "side", "san", "cpl", "move_class", "clock_s")},
                "uci": move["uci"],
            }
            for row, move in zip(rows, game["moves"])
        ],
        influential_moves=influence,
        analysis=dict(source=analysis["analysis_source"], nodes=engine.nodes, seconds=seconds),
        model={k: metadata[k] for k in ("name", "version", "run_id")},
        warnings=warnings,
    )


def trainer(path, payload=None, headers=None):
    """Proxy a trainer request and decode its JSON response (DESIGN D9)."""
    return remote(
        os.getenv("TRAINER_URL", "http://trainer:8001") + path,
        payload=payload,
        code="trainer_unavailable",
        headers=headers,
    ).json()


@app.get("/v1/pipeline/dag")
def dag():
    """Return the trainer-provided Mermaid pipeline DAG (DESIGN §9)."""
    return trainer("/dag")


@app.get("/v1/pipeline/runs")
def runs(limit: int = 100):
    """Return trainer history with a bounded requested row count (DESIGN §9)."""
    return trainer(f"/runs?limit={max(1, min(limit, 1000))}")


@app.get("/v1/pipeline/replay")
def replay():
    """Read-only, bounded monitor projection; never advances the replay clock."""
    status = remote(
        os.getenv("MONITOR_URL", "http://monitor:8002").rstrip("/") + "/status",
        code="monitor_unavailable",
    ).json()
    quality = status.get("last_metrics", {})
    mae, reference = quality.get("mae", {}), quality.get("reference_mae")
    cfg = load_params()["monitor"]
    champion = mae.get("champion", {}).get("all")
    return dict(
        current_month=status.get("current_month"),
        months_revealed=len(status.get("revealed", [])),
        total_months=12,
        playing=status.get("playing", False),
        seconds_to_next_reveal=status.get("seconds_to_next_reveal"),
        champion_version=status.get("champion_version"),
        previous_champion_version=status.get("previous_champion_version"),
        metrics_month=quality.get("month"),
        reference_mae=reference,
        mae={key: value.get("all") for key, value in mae.items()},
        mae_ratio={
            key: value.get("all") / reference if reference else None for key, value in mae.items()
        },
        coverage=quality.get("coverage", {}).get("all"),
        drift_detected=(
            (
                champion > cfg["mae_ratio"] * reference
                or max(quality.get("psi", {}).values(), default=0) > cfg["psi_warn"]
            )
            if champion is not None and reference
            else None
        ),
        drift_injection=status.get("drift_injection", {}).get("enabled", False),
        psi_top5=status.get("psi_top5", [])[:5],
    )


@app.post("/v1/pipeline/retrain", status_code=202)
def retrain(request: Request):
    """Authenticate and forward a manual retrain request (DESIGN D20, §24 F1).

    Args:
        request: Request with an X-API-Key and optional JSON reason.

    Returns:
        The trainer run identifier on acceptance.

    Raises:
        ApiError: Authentication, JSON decoding or upstream submission fails,
            including the trainer's 409 when another run is active."""
    import hmac

    expected = os.getenv("API_KEY", "")
    if not expected or not hmac.compare_digest(request.headers.get("X-API-Key", ""), expected):
        raise ApiError("unauthorized", "Invalid API key", 401)
    try:
        payload = json.loads(request.state.body or b"{}")
    except ValueError as exc:
        raise ApiError("validation_error", "Invalid JSON") from exc
    return trainer("/retrain", payload, headers={"X-API-Key": expected})
