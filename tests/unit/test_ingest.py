"""Exercise the monthly data contract with local Spark and no network."""

import hashlib
import io
import sys
from pathlib import Path

import pytest
import zstandard
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from elo.common.config import load_params
from elo.common.errors import DataValidationError
from elo.data.download import copy_games
from elo.data.spark_ingest import parse_records, tc_class, validate
from elo.data.spark_sample import sample_games

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/pgn/mini.pgn"


@pytest.fixture(scope="module")
def spark():
    """One core and a small heap keep tests suitable for this laptop."""
    session = (
        SparkSession.builder.master("local[1]")
        .appName("test-ingest")
        .config("spark.driver.memory", "512m")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.pyspark.python", sys.executable)
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="module")
def parsed(spark):
    records = spark.read.option("lineSep", "\n\n[Event ").text(str(FIXTURE))
    frame = parse_records(records, "2019-09", load_params()).cache()
    yield frame
    frame.unpersist()


def test_headers(parsed):
    row = parsed.filter(F.col("game_id") == "Rapid001").first()
    assert row.event == "Rated Rapid game"
    assert row.utc_date == "2019.09.01"
    assert row.tc_base == 600 and row.tc_inc == 0
    assert row.n_plies == 20 and row.has_clock and row.has_eval
    identity = f"{load_params()['seed']}AlicePlayer".encode()
    assert row.white_hash == hashlib.sha256(identity).hexdigest()[:16]
    assert row.role == "train"


def test_boundaries(spark):
    values = [0, 28, 29, 178, 179, 478, 479, 1498, 1499, None]
    output = spark.createDataFrame([(v,) for v in values], "estimate int")
    labels = [
        r.label for r in output.select(tc_class(F.col("estimate")).alias("label")).toLocalIterator()
    ]
    assert labels == [
        "ultrabullet",
        "ultrabullet",
        "bullet",
        "bullet",
        "blitz",
        "blitz",
        "rapid",
        "rapid",
        "classical",
        None,
    ]


def test_filters_schema_leakage_and_sampling(parsed):
    output, report = validate(parsed)
    try:
        assert report["raw_games"] == 8
        assert report["removed_by_filter"] == dict(
            rated=1, standard=1, clock=1, time_control=1, plies=1, elo=0, not_aborted=0
        )
        assert report["duplicates"] == 1 and report["kept"] == 2
        expected = (
            "game_id:string month:string role:string utc_date:string event:string "
            "time_control:string tc_base:int tc_inc:int tc_class:string result:string "
            "termination:string eco:string opening:string white_hash:string black_hash:string "
            "n_plies:int has_clock:boolean has_eval:boolean movetext:string "
            "label_white_elo:int label_black_elo:int"
        )
        assert " ".join(f"{f.name}:{f.dataType.simpleString()}" for f in output.schema) == expected
        for row in output.toLocalIterator():
            for forbidden in ("Elo", "RatingDiff", "Title", "AlicePlayer", "BobPlayer"):
                assert forbidden not in row.movetext
                assert not any(forbidden in column for column in output.columns)
        # More rows exercise nontrivial seeded Bernoulli selection, including trim.
        expanded = (
            output.crossJoin(output.sparkSession.range(100))
            .withColumn("game_id", F.concat("game_id", F.col("id")))
            .drop("id")
        )
        first = [r.game_id for r in sample_games(expanded, 80, 42).toLocalIterator()]
        second = [r.game_id for r in sample_games(expanded, 80, 42).toLocalIterator()]
        assert first == second and len(first) == 80
    finally:
        output.unpersist()


def test_invalid_ratings_aborted_and_correspondence(parsed):
    valid = parsed.filter(F.col("game_id") == "Rapid001")
    bad_elo = valid.withColumn("label_white_elo", F.lit(599))
    aborted = valid.withColumn("termination", F.lit("Aborted"))
    correspondence = valid.withColumn("tc_class", F.lit(None).cast("string"))
    output, report = validate(valid.union(bad_elo).union(aborted).union(correspondence))
    assert report["removed_by_filter"]["elo"] == 1
    assert report["removed_by_filter"]["not_aborted"] == 1
    assert report["removed_by_filter"]["time_control"] == 1
    output.unpersist()
    with pytest.raises(DataValidationError, match="No games"):
        validate(bad_elo)
    with pytest.raises(DataValidationError, match="Required columns"):
        validate(valid.withColumn("utc_date", F.lit(None).cast("string")))


@pytest.mark.parametrize("maximum,expected", [(1, 1), (3, 3), (8, 8), (20, 8)])
def test_stream_boundaries(maximum, expected):
    data = FIXTURE.read_bytes()
    compressed = zstandard.ZstdCompressor().compress(data)
    destination = io.BytesIO()
    games, size = copy_games(io.BytesIO(compressed), destination, maximum)
    result = destination.getvalue()
    assert games == expected and size == len(result)
    chunks = data.split(b"\n\n[Event ")
    assert result == b"\n\n[Event ".join(chunks[:expected]) + (b"\n\n" if expected < 8 else b"")
