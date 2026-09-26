from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


def csv_members(path: Path) -> list[str]:
    if path.suffix.lower() != ".zip":
        return [str(path)]
    with zipfile.ZipFile(path) as archive:
        return sorted(
            n for n in archive.namelist()
            if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")
        )


def _read_header(path: Path, member: str | None = None) -> list[str]:
    if member is None:
        return list(pd.read_csv(path, nrows=0).columns)
    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        return list(pd.read_csv(stream, nrows=0).columns)


def combine_csvs(
    inputs: Iterable[Path], output: Path, timestamp: str, chunksize: int = 250_000
) -> dict:
    """Chronologically combine CSV/ZIP exports with header and timestamp checks.

    This is streaming and suitable for multi-GB Pason archives. Duplicate
    timestamps are removed after a stable chronological merge.
    """
    sources: list[tuple[Path, str | None]] = []
    for path in inputs:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".zip":
            sources.extend((path, m) for m in csv_members(path))
        else:
            sources.append((path, None))
    if not sources:
        raise ValueError("No CSV files found")
    expected = _read_header(*sources[0])
    if timestamp not in expected:
        raise ValueError(f"Timestamp column {timestamp!r} not found; columns: {expected}")
    frames = []
    rows_read = 0
    for path, member in sources:
        columns = _read_header(path, member)
        if columns != expected:
            raise ValueError(f"Header mismatch in {member or path}: {columns}")
        if member is None:
            iterator = pd.read_csv(path, chunksize=chunksize, low_memory=False)
            for chunk in iterator:
                rows_read += len(chunk); frames.append(chunk)
        else:
            with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
                for chunk in pd.read_csv(stream, chunksize=chunksize, low_memory=False):
                    rows_read += len(chunk); frames.append(chunk)
    data = pd.concat(frames, ignore_index=True)
    data["__timestamp"] = pd.to_datetime(data[timestamp], errors="coerce", utc=True)
    invalid = int(data["__timestamp"].isna().sum())
    data = data.dropna(subset=["__timestamp"]).sort_values("__timestamp", kind="stable")
    before = len(data)
    data = data.drop_duplicates("__timestamp", keep="last").drop(columns="__timestamp")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, quoting=csv.QUOTE_MINIMAL)
    return {"sources": len(sources), "rows_read": rows_read, "invalid_timestamps": invalid,
            "duplicate_timestamps_removed": before - len(data), "rows_written": len(data)}

