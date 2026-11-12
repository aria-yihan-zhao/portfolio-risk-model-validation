"""Download, parse, and validate the five-industry daily return dataset."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SOURCE_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "5_Industry_Portfolios_Daily_CSV.zip"
)
EXPECTED_COLUMNS = ["Cnsmr", "Manuf", "HiTec", "Hlth", "Other"]
DATE_PATTERN = re.compile(r"^\d{8}$")


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of an in-memory payload."""

    return hashlib.sha256(payload).hexdigest()


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a stable, human-readable JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_dataset(
    destination: Path,
    manifest_path: Path,
    url: str = SOURCE_URL,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Download the official ZIP, extract its CSV, and record provenance."""

    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    payload = response.content
    digest = sha256_bytes(payload)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one CSV in archive, found {members}")
        csv_payload = archive.read(members[0])

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(csv_payload)
    frame = load_industry_returns(destination)
    manifest = {
        "source_url": url,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_sha256": digest,
        "archive_bytes": len(payload),
        "csv_bytes": len(csv_payload),
        "destination": f"data/raw/{destination.name}",
        "columns": EXPECTED_COLUMNS,
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
        "rows": int(len(frame)),
    }
    write_json(manifest, manifest_path)
    return manifest


def load_industry_returns(
    path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Parse the value-weighted daily section and return decimal returns."""

    lines = path.read_text(encoding="utf-8-sig", errors="strict").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.strip().startswith(",Cnsmr")),
        None,
    )
    if header_index is None:
        raise ValueError("Could not locate the value-weighted daily return header")

    records: list[list[str]] = []
    for line in lines[header_index + 1 :]:
        row = next(csv.reader([line]))
        if not row or not DATE_PATTERN.fullmatch(row[0].strip()):
            if records:
                break
            continue
        records.append([value.strip() for value in row])

    if not records:
        raise ValueError("No daily return observations found")
    frame = pd.DataFrame(records, columns=["date", *EXPECTED_COLUMNS])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d", errors="raise")
    for column in EXPECTED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise") / 100.0
    frame = frame.set_index("date").sort_index()
    frame = frame.replace({-0.9999: pd.NA, -9.99: pd.NA}).dropna()
    frame = frame.astype(float)

    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and sorted")
    if frame.abs().max().max() > 1.0:
        raise ValueError("Returns appear not to be expressed in decimal units")
    if start_date is not None:
        frame = frame.loc[pd.Timestamp(start_date) :]
    if end_date is not None:
        frame = frame.loc[: pd.Timestamp(end_date)]
    if frame.empty:
        raise ValueError("Date filters produced an empty dataset")
    return frame


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Combine asset returns using fixed weights after strict alignment checks."""

    if list(asset_returns.columns) != list(weights.index):
        raise ValueError("Weight labels and return columns must match in the same order")
    if not abs(float(weights.sum()) - 1.0) < 1e-12:
        raise ValueError("Portfolio weights must sum to one")
    if (weights < 0).any():
        raise ValueError("This project uses a long-only portfolio")
    result = asset_returns.mul(weights, axis="columns").sum(axis="columns")
    result.name = "portfolio_return"
    return result
