"""Parse, filter and validate monthly PGN records with Spark DataFrames."""

import argparse
import io
import json
import logging
import sys
import time
from contextlib import redirect_stdout
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from elo.common.config import load_params
from elo.common.errors import DataValidationError
from elo.common.logging import get_logger

LOG = get_logger(__name__)
CONTRACT = {
    "game_id": "string",
    "month": "string",
    "role": "string",
    "utc_date": "string",
    "event": "string",
    "time_control": "string",
    "tc_base": "int",
    "tc_inc": "int",
    "tc_class": "string",
    "result": "string",
    "termination": "string",
    "eco": "string",
    "opening": "string",
    "white_hash": "string",
    "black_hash": "string",
    "n_plies": "int",
    "has_clock": "boolean",
    "has_eval": "boolean",
    "movetext": "string",
    "label_white_elo": "int",
    "label_black_elo": "int",
}


def session(params: dict) -> SparkSession:
    """Keep the local Spark UI available while each stage runs."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("elo-data")
        .config("spark.driver.memory", params["spark"]["driver_memory"])
        .config("spark.sql.shuffle.partitions", params["spark"]["shuffle_partitions"])
        .config("spark.ui.enabled", "true")
        .config("spark.ui.port", "4040")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.hadoop.textinputformat.record.delimiter", "\n\n[Event ")
        .getOrCreate()
    )


def tc_class(estimate):
    """Lichess time-control boundaries use base + 40 increments."""
    return (
        F.when(estimate < 29, "ultrabullet")
        .when(estimate < 179, "bullet")
        .when(estimate < 479, "blitz")
        .when(estimate < 1499, "rapid")
        .when(estimate.isNotNull(), "classical")
    )


def parse_records(records: DataFrame, month: str, params: dict) -> DataFrame:
    """Extract headers separately so identities and rating headers cannot enter movetext."""

    def header(name):
        return F.regexp_extract("headers", rf'(?m)^\[{name} "([^"\r\n]*)"\]\s*$', 1)

    frame = records.select(F.regexp_replace("value", r"^(?!\[Event )", "[Event ").alias("pgn"))
    frame = frame.withColumn("headers", F.regexp_extract("pgn", r"(?s)^(.*?)\r?\n\r?\n", 1))
    frame = frame.withColumn("movetext", F.regexp_replace("pgn", r"(?s)^.*?\r?\n\r?\n", ""))
    for name, key in {
        "event": "Event",
        "time_control": "TimeControl",
        "utc_date": "UTCDate",
        "result": "Result",
        "termination": "Termination",
        "eco": "ECO",
        "opening": "Opening",
        "variant": "Variant",
    }.items():
        frame = frame.withColumn(name, header(key))
    frame = frame.withColumn(
        "game_id", F.regexp_extract(header("Site"), r"https?://lichess.org/([A-Za-z0-9]+)", 1)
    )
    for side in ("white", "black"):
        frame = frame.withColumn(f"label_{side}_elo", header(f"{side.title()}Elo").try_cast("int"))
        frame = frame.withColumn(
            f"{side}_hash",
            F.substring(
                F.sha2(F.concat(F.lit(str(params["seed"])), header(side.title())), 256), 1, 16
            ),
        )
    for key, group in (("tc_base", 1), ("tc_inc", 2)):
        frame = frame.withColumn(
            key, F.regexp_extract("time_control", r"^(\d+)\+(\d+)$", group).try_cast("int")
        )
    # Ignore comments, variations, NAGs, results and move numbers before counting SAN.
    moves = F.regexp_replace("movetext", r"(?s)\{.*?\}|;[^\r\n]*|\([^()]*\)|\$\d+", " ")
    moves = F.regexp_replace(moves, r"\d+\.(?:\.\.)?|1-0|0-1|1/2-1/2|\*", " ")
    frame = frame.withColumn(
        "n_plies",
        F.size(
            F.regexp_extract_all(
                moves,
                F.lit(r"(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#?!]*"),
                0,
            )
        ),
    )
    return (
        frame.withColumn("tc_class", tc_class(F.col("tc_base") + 40 * F.col("tc_inc")))
        .withColumn("has_clock", F.col("movetext").contains("%clk"))
        .withColumn("has_eval", F.col("movetext").contains("%eval"))
        .withColumn("month", F.lit(month))
        .withColumn("role", F.lit("train" if month in params["data"]["train_months"] else "replay"))
    )


def validate(frame: DataFrame) -> tuple[DataFrame, dict]:
    """Count removals sequentially; aggregate only bounded summaries on the driver."""
    raw = frame.count()
    filters = {
        "rated": F.col("event").rlike(r"(?i)^rated\b"),
        "standard": F.col("variant").isin("", "Standard"),
        "clock": F.col("has_clock"),
        "time_control": F.col("tc_class").isin("blitz", "rapid", "classical"),
        "plies": F.col("n_plies") >= 20,
        "elo": F.col("label_white_elo").between(600, 3200)
        & F.col("label_black_elo").between(600, 3200),
        "not_aborted": (F.col("termination") != "Aborted")
        & F.col("result").isin("1-0", "0-1", "1/2-1/2"),
    }
    # One aggregate computes the same sequential counts without rerunning the parser per filter.
    predicates, cumulative = [], F.lit(True)
    for name, predicate in filters.items():
        cumulative = cumulative & F.coalesce(predicate, F.lit(False))
        predicates.append(F.sum(cumulative.cast("long")).alias(name))
    counts = frame.agg(*predicates).first().asDict()
    removed, previous = {}, raw
    for name, count in counts.items():
        count = count or 0
        removed[name], previous = previous - count, count
    output = frame.filter(cumulative).select(*CONTRACT).dropDuplicates(["game_id"]).cache()
    kept = output.count()
    if not kept:
        output.unpersist()
        raise DataValidationError("No games remain after validation")
    required = reduce(lambda a, b: a | b, [F.col(c).isNull() for c in CONTRACT])
    if (
        output.filter(required | (F.col("game_id") == "") | (F.col("utc_date") == ""))
        .limit(1)
        .count()
    ):
        output.unpersist()
        raise DataValidationError("Required columns contain null or missing values")
    ratings = output.select(F.col("label_white_elo").alias("elo")).union(
        output.select("label_black_elo")
    )
    stats = (
        ratings.agg(F.mean("elo").alias("elo_mean"), F.stddev("elo").alias("elo_std"))
        .first()
        .asDict()
    )
    distribution = {
        r.tc_class: r["count"] for r in output.groupBy("tc_class").count().toLocalIterator()
    }
    return output, dict(
        raw_games=raw,
        removed_by_filter=removed,
        duplicates=previous - kept,
        kept=kept,
        **stats,
        tc_class=distribution,
    )


def write_report(path: str, report: dict) -> None:
    """Publish a JSON metric report."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")


def ingest(spark: SparkSession, month: str, params: dict) -> dict:
    """Read records with Hadoop's delimiter and persist only the safe contract."""
    started = time.monotonic()
    records = spark.read.option("lineSep", "\n\n[Event ").text(f"data/raw/{month}.pgn")
    parsed = parse_records(records, month, params)
    if LOG.isEnabledFor(logging.DEBUG):
        plan = io.StringIO()
        with redirect_stdout(plan):
            parsed.explain()
        LOG.debug("Spark plan: %s", plan.getvalue())
    output, report = validate(parsed)
    try:
        output.write.mode("overwrite").parquet(f"data/interim/games/{month}.parquet")
        report["seconds"] = time.monotonic() - started
        report["games_per_second"] = report["raw_games"] / report["seconds"]
        write_report(f"reports/validation/{month}.json", report)
        LOG.info("Validated %s: %s", month, report)
        return report
    finally:
        output.unpersist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True)
    args = parser.parse_args()
    params = load_params()
    spark = session(params)
    try:
        ingest(spark, args.month, params)
    finally:
        spark.stop()
