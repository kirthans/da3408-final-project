"""Stream only the first bounded set of games from a monthly Lichess dump."""

import argparse
import io
import json
import time
from pathlib import Path
from typing import BinaryIO

import requests
import zstandard
from urllib3.exceptions import HTTPError

from elo.common.config import load_params
from elo.common.errors import DataValidationError
from elo.common.logging import get_logger

LOG = get_logger(__name__)


def copy_games(source: BinaryIO, destination: BinaryIO, maximum: int) -> tuple[int, int]:
    """Stream a bounded number of complete games from a compressed dump.

    Args:
        source: Binary Zstandard-compressed input stream.
        destination: Binary destination receiving decompressed UTF-8 PGN.
        maximum: Maximum number of Event-delimited games to copy.

    Returns:
        Counts of games and decompressed bytes written. The next Event header
        stops copying so the last selected game remains complete."""
    games = size = 0
    with zstandard.ZstdDecompressor().stream_reader(source) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8") as lines:
            for line in lines:
                if line.startswith("[Event "):
                    if games == maximum:
                        break
                    games += 1
                if games:
                    chunk = line.encode("utf-8")
                    destination.write(chunk)
                    size += len(chunk)
    return games, size


def download(month: str, maximum: int, base_url: str) -> dict:
    """Retry a bounded monthly download and atomically publish complete output.

    Args:
        month: Lichess partition name in YYYY-MM format.
        maximum: Positive cap on the number of complete games.
        base_url: Root URL of the monthly Lichess dumps.

    Returns:
        Game/byte counts, elapsed seconds and decompressed throughput.

    Raises:
        ValueError: The game cap is not positive.
        DataValidationError: The downloaded stream contains no games.
        requests.RequestException: HTTP attempts are exhausted.
        OSError: Stream or filesystem attempts are exhausted.
        zstandard.ZstdError: Decompression attempts are exhausted."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    output = Path(f"data/raw/{month}.pgn")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".pgn.part")
    started = time.monotonic()
    url = f"{base_url}/lichess_db_standard_rated_{month}.pgn.zst"
    try:
        for attempt in range(4):
            try:
                with requests.get(url, stream=True, timeout=(15, 90)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as stream:
                        games, size = copy_games(response.raw, stream, maximum)
                if not games:
                    raise DataValidationError("Downloaded stream contains no games")
                temporary.replace(output)
                break
            except (requests.RequestException, HTTPError, OSError, zstandard.ZstdError):
                if attempt == 3:
                    raise
                LOG.warning("Download %s failed; retry %d", month, attempt + 1, exc_info=True)
                time.sleep(2**attempt)
    finally:
        temporary.unlink(missing_ok=True)
    seconds = time.monotonic() - started
    report = dict(games=games, bytes=size, seconds=seconds, mb_per_second=size / 1e6 / seconds)
    path = Path(f"reports/download/{month}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    LOG.info("Downloaded %s: %s (bytes and MB/s measure decompressed PGN)", month, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True)
    args = parser.parse_args()
    params = load_params()["data"]
    download(args.month, params["max_games_per_month"], params["base_url"])
