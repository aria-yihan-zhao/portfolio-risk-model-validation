#!/usr/bin/env python3
"""Download and verify the official five-industry daily dataset."""

from pathlib import Path

from portfolio_risk.data import download_dataset

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = download_dataset(
        ROOT / "data" / "raw" / "5_Industry_Portfolios_Daily.csv",
        ROOT / "reports" / "tables" / "data_manifest.json",
    )
    print(f"Downloaded {manifest['rows']:,} observations through {manifest['last_date']}.")
    print(f"Archive SHA-256: {manifest['archive_sha256']}")


if __name__ == "__main__":
    main()
