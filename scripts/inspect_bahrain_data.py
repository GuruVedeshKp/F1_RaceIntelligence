from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openf1"
    / "2024"
    / "bahrain_9472"
)

REPORT_DIR = PROJECT_ROOT / "data" / "processed"

ENDPOINTS = [
    "sessions",
    "drivers",
    "laps",
    "stints",
    "pit",
    "position",
    "intervals",
    "race_control",
    "weather",
    "session_result",
]

# Natural keys expected to identify unique records.
UNIQUE_KEYS = {
    "drivers": ["driver_number"],
    "laps": ["driver_number", "lap_number"],
    "stints": ["driver_number", "stint_number"],
    "session_result": ["driver_number"],
}


def inspect_endpoint(endpoint: str) -> list[str]:
    """Inspect one Parquet dataset and return report lines."""

    path = DATA_DIR / f"{endpoint}.parquet"
    lines = []

    lines.append("=" * 65)
    lines.append(f"DATASET: {endpoint.upper()}")
    lines.append("=" * 65)

    if not path.exists():
        lines.append("ERROR: Parquet file not found.")
        return lines

    df = pd.read_parquet(path)

    lines.append(f"Records: {len(df)}")
    lines.append(f"Columns: {len(df.columns)}")
    lines.append(f"Duplicate full rows: {df.duplicated().sum()}")

    # Validate expected unique keys.
    if endpoint in UNIQUE_KEYS:
        keys = UNIQUE_KEYS[endpoint]

        missing_keys = [
            column for column in keys
            if column not in df.columns
        ]

        if missing_keys:
            lines.append(
                f"KEY CHECK: Missing columns {missing_keys}"
            )
        else:
            duplicates = df.duplicated(subset=keys).sum()
            lines.append(
                f"Duplicate natural keys ({keys}): {duplicates}"
            )

    # Check session consistency.
    if "session_key" in df.columns:
        session_keys = df["session_key"].dropna().unique()
        lines.append(
            f"Session keys: {list(session_keys)}"
        )

    # Column-level information.
    lines.append("\nCOLUMN DETAILS:")

    for column in df.columns:
        missing = int(df[column].isna().sum())
        percentage = (
            missing / len(df) * 100
            if len(df) > 0
            else 0
        )

        lines.append(
            f"{column}: "
            f"dtype={df[column].dtype}, "
            f"missing={missing} ({percentage:.2f}%)"
        )

    # Sample records.
    lines.append("\nSAMPLE RECORDS:")
    lines.append(df.head(3).to_string(index=False))

    return lines


def main() -> None:
    """Generate a data quality report for Bahrain 2024."""

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    all_lines = [
        "F1 RACE INTELLIGENCE",
        "2024 BAHRAIN GRAND PRIX - DATA QUALITY REPORT",
        "",
    ]

    total_records = 0

    for endpoint in ENDPOINTS:
        print(f"Inspecting {endpoint}...")

        lines = inspect_endpoint(endpoint)
        all_lines.extend(lines)

        path = DATA_DIR / f"{endpoint}.parquet"

        if path.exists():
            total_records += len(pd.read_parquet(path))

    all_lines.extend([
        "",
        "=" * 65,
        f"TOTAL RECORDS: {total_records}",
        "=" * 65,
    ])

    report_path = REPORT_DIR / "bahrain_2024_data_quality.txt"

    report_path.write_text(
        "\n".join(all_lines),
        encoding="utf-8",
    )

    print("\nData quality inspection completed!")
    print(f"Total records: {total_records}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
