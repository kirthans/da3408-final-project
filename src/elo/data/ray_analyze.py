"""Month-resumable Ray actor pool for shared per-move analysis."""

import argparse
import json
import time
from pathlib import Path

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import ray

from elo.common.config import load_params
from elo.common.errors import EloError
from elo.common.logging import get_logger
from elo.engine import EngineAnalyzer
from elo.features.moves import build_move_rows
from elo.pgn import PgnParser

logger = get_logger(__name__)
MOVE_SCHEMA = pa.schema(
    [("game_id", pa.string()), ("ply", pa.int32()), ("side", pa.string()), ("san", pa.string())]
    + [(name, pa.float64()) for name in ("eval_before_cp", "eval_after_cp", "cpl", "wp_loss")]
    + [
        ("is_best", pa.bool_()),
        ("move_class", pa.string()),
        ("legal_moves", pa.int32()),
        ("in_check", pa.bool_()),
        ("material_mover", pa.int32()),
        ("material_change", pa.int32()),
    ]
    + [(name, pa.float64()) for name in ("clock_s", "time_spent_s", "clock_frac")]
    + [("time_trouble", pa.bool_()), ("phase", pa.string())]
)
BATCH_SCHEMA = MOVE_SCHEMA.append(pa.field("_status", pa.string()))


class AnalyzerActor:
    """Construct one persistent Stockfish process per Ray actor."""

    def __init__(self, engine: dict, log_every: int = 10):
        self.analyzer = EngineAnalyzer(**engine)
        self.parser = PgnParser()
        self.batches, self.log_every = 0, log_every

    def __call__(self, batch: pa.Table) -> pa.Table:
        rows = []
        for game in batch.to_pylist():
            try:
                moves = self.parser.parse_movetext(game["movetext"])
                analysis = self.analyzer.analyse_game(chess.Board(), moves)
                game_rows = build_move_rows(game, moves, analysis)
                for row in game_rows:
                    row["_status"] = "move"
                game_rows[0]["_status"] = analysis["analysis_source"]
                rows.extend(game_rows)
            except (EloError, ValueError, KeyError) as exc:
                logger.warning("Skipping game_id=%s: %s", game.get("game_id"), exc)
                rows.append({"game_id": game.get("game_id"), "_status": "failed"})
        self.batches += 1
        if self.batches % self.log_every == 0:
            logger.info("Completed %s batches in actor", self.batches)
        return pa.Table.from_pylist(rows, schema=BATCH_SCHEMA)

    def __del__(self):
        if hasattr(self, "analyzer"):
            self.analyzer.close()


def run(month: str, params_path: str = "params.yaml") -> dict:
    """Analyze a month and report measured end-to-end throughput including startup."""
    params = load_params(params_path)
    workers = params["ray"]["num_workers"]
    start = time.monotonic()
    games = plies = failed = reused = 0
    output = Path(f"data/features/moves/{month}.parquet")
    report = Path(f"reports/throughput/analyze_{month}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    ray.init(num_cpus=workers, include_dashboard=True, dashboard_port=8265)
    try:
        # Finish the small sampled input before reserving every CPU for actors.
        # Ray 2.58 otherwise starves its lazy ListFiles/read tasks of a CPU.
        dataset = (
            ray.data.read_parquet(f"data/interim/sample/{month}.parquet")
            .repartition(workers * 4)
            .materialize()
        )
        analyzed = dataset.map_batches(
            AnalyzerActor,
            fn_constructor_kwargs={"engine": params["engine"]},
            concurrency=workers,
            num_cpus=1,
            batch_size=params["ray"]["batch_size"],
            batch_format="pyarrow",
        )
        # Stream batches to a fixed schema: even all-null clock fields stay typed.
        with pq.ParquetWriter(output, MOVE_SCHEMA) as writer:
            for number, batch in enumerate(analyzed.iter_batches(batch_format="pyarrow"), 1):
                statuses = batch["_status"].to_pylist()
                failed += statuses.count("failed")
                reused += statuses.count("pgn_eval")
                games += statuses.count("pgn_eval") + statuses.count("stockfish")
                clean = batch.filter(pa.array([s != "failed" for s in statuses]))
                clean = clean.select(MOVE_SCHEMA.names)
                plies += clean.num_rows
                writer.write_table(clean)
                if number % 10 == 0:
                    logger.info(
                        "Written batches=%s games=%s plies=%s failed=%s",
                        number,
                        games,
                        plies,
                        failed,
                    )
        seconds = time.monotonic() - start
        metrics = dict(
            games=games,
            plies=plies,
            failed_games=failed,
            seconds=seconds,
            games_per_second=games / seconds,
            plies_per_second=plies / seconds,
            num_workers=workers,
            nodes=params["engine"]["nodes"],
            pgn_eval_fraction=reused / games if games else 0,
        )
        report.write_text(json.dumps(metrics, indent=2) + "\n")
        logger.info("Analysis complete: %s", metrics)
        return metrics
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    run(args.month, args.params)
