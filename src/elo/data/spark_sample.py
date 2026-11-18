"""Seeded proportional sampling of validated monthly games."""

import argparse

from pyspark.sql import DataFrame

from elo.common.config import load_params
from elo.common.logging import get_logger
from elo.data.spark_ingest import CONTRACT, session, write_report

LOG = get_logger(__name__)


def sample_games(frame: DataFrame, target: int, seed: int) -> DataFrame:
    """Take a seeded time-control sample with stable input ordering.

    Args:
        frame: Nonempty validated monthly game dataframe.
        target: Maximum output games; a 5% oversample reduces short strata draws.
        seed: Shared sampling seed.

    Returns:
        Safe contract columns from a stratified draw capped at target. Stable
        game ordering makes the draw insensitive to Parquet file layout."""
    counts = {
        row.tc_class: row["count"] for row in frame.groupBy("tc_class").count().toLocalIterator()
    }
    total = sum(counts.values())
    fraction = min(1.0, 1.05 * target / total)
    return (
        frame.orderBy("game_id")
        .sampleBy("tc_class", {key: fraction for key in counts}, seed)
        .limit(target)
        .select(*CONTRACT)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True)
    args = parser.parse_args()
    params = load_params()
    role = "train" if args.month in params["data"]["train_months"] else "replay"
    spark = session(params)
    try:
        frame = spark.read.parquet(f"data/interim/games/{args.month}.parquet")
        output = sample_games(frame, params["sample"][f"{role}_per_month"], params["seed"]).cache()
        counts = {
            r.tc_class: r["count"] for r in output.groupBy("tc_class").count().toLocalIterator()
        }
        if sum(counts.values()) < params["sample"][f"{role}_per_month"]:
            LOG.warning("Sample %s below target: %s", args.month, counts)
        output.write.mode("overwrite").parquet(f"data/interim/sample/{args.month}.parquet")
        write_report(f"reports/sample/{args.month}.json", counts)
        LOG.info("Sampled %s: %s", args.month, counts)
    finally:
        spark.stop()
