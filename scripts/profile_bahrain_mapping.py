from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data" / "raw" / "openf1" / "2024" / "bahrain_9472"
)

OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = OUTPUT_DIR / "bahrain_2024_mapping_audit.txt"


def load_dataset(name: str) -> pd.DataFrame:
    """Load one raw Parquet dataset."""
    path = DATA_DIR / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    return pd.read_parquet(path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def write(title: str, frame: pd.DataFrame) -> None:
        lines.append("\n" + "=" * 75)
        lines.append(title)
        lines.append("=" * 75)
        lines.append(frame.to_string(index=False))

    # ------------------------------------------------
    # 1. DRIVER AND LAP COVERAGE
    # ------------------------------------------------

    drivers = load_dataset("drivers")
    laps = load_dataset("laps")
    results = load_dataset("session_result")

    lap_counts = (
        laps.groupby("driver_number")
        .agg(
            lap_records=("lap_number", "size"),
            first_lap=("lap_number", "min"),
            last_lap=("lap_number", "max"),
        )
        .reset_index()
    )

    coverage = (
        drivers[
            ["driver_number", "name_acronym", "full_name"]
        ]
        .merge(lap_counts, on="driver_number", how="left")
        .merge(
            results[
                ["driver_number", "number_of_laps",
                 "position", "dnf", "dns", "dsq"]
            ],
            on="driver_number",
            how="left",
        )
    )

    missing_laps = []

    for _, row in coverage.iterrows():

        if pd.isna(row["number_of_laps"]):
            missing_laps.append("RESULT MISSING")
            continue

        observed = set(
            laps.loc[
                laps["driver_number"] == row["driver_number"],
                "lap_number"
            ].dropna().astype(int)
        )

        expected = set(
            range(1, int(row["number_of_laps"]) + 1)
        )

        missing = sorted(expected - observed)

        missing_laps.append(
            ",".join(map(str, missing)) if missing else "None"
        )

    coverage["missing_lap_numbers"] = missing_laps

    write("1. DRIVER LAP COVERAGE", coverage)

    # ------------------------------------------------
    # 2. PIT STOP INSPECTION
    # ------------------------------------------------

    pit = load_dataset("pit")

    pit_columns = [
        column for column in [
            "driver_number",
            "lap_number",
            "date",
            "lane_duration",
            "stop_duration",
        ]
        if column in pit.columns
    ]

    write(
        "2. PIT STOP RECORDS",
        pit[pit_columns].sort_values(
            ["driver_number", "lap_number"]
        ),
    )

    # ------------------------------------------------
    # 3. RACE CONTROL EVENTS
    # ------------------------------------------------

    race_control = load_dataset("race_control")

    category_counts = (
        race_control.groupby(
            ["category", "flag"],
            dropna=False,
        )
        .size()
        .reset_index(name="record_count")
        .sort_values("record_count", ascending=False)
    )

    write("3. RACE CONTROL CATEGORIES", category_counts)

    relevant = race_control[
        race_control["message"].str.contains(
            r"SAFETY CAR|VIRTUAL SAFETY CAR|"
            r"YELLOW|RED FLAG|CHEQUERED|GREEN FLAG",
            case=False,
            na=False,
        )
    ]

    event_columns = [
        column for column in [
            "date",
            "lap_number",
            "category",
            "flag",
            "scope",
            "message",
        ]
        if column in relevant.columns
    ]

    write(
        "4. SAFETY CAR AND FLAG MESSAGES",
        relevant[event_columns].sort_values("date"),
    )

    # ------------------------------------------------
    # 4. POSITION DATA COVERAGE
    # ------------------------------------------------

    position = load_dataset("position")

    position["date"] = pd.to_datetime(
        position["date"],
        format="ISO8601",
        utc=True,
    )

    position_summary = (
        position.groupby("driver_number")
        .agg(
            position_records=("position", "size"),
            first_update=("date", "min"),
            last_update=("date", "max"),
        )
        .reset_index()
    )

    write("5. POSITION UPDATE COVERAGE", position_summary)

    # ------------------------------------------------
    # 5. WEATHER COVERAGE
    # ------------------------------------------------

    weather = load_dataset("weather")

    weather["date"] = pd.to_datetime(
        weather["date"],
        format="ISO8601",
        utc=True,
    )

    weather = weather.sort_values("date")

    weather_gaps = weather["date"].diff().dt.total_seconds()

    weather_summary = pd.DataFrame({
        "metric": [
            "Number of readings",
            "First reading",
            "Last reading",
            "Median interval (seconds)",
            "Maximum interval (seconds)",
        ],
        "value": [
            len(weather),
            str(weather["date"].min()),
            str(weather["date"].max()),
            round(weather_gaps.median(), 2),
            round(weather_gaps.max(), 2),
        ],
    })

    write("6. WEATHER COVERAGE", weather_summary)

    # ------------------------------------------------
    # 6. INTERVAL GAP FORMAT
    # ------------------------------------------------

    intervals = load_dataset("intervals")

    gap_strings = (
        intervals["gap_to_leader"]
        .dropna()
        .astype(str)
    )

    lapped_records = gap_strings[
        gap_strings.str.contains(
            r"\+.*LAP",
            case=False,
            regex=True,
        )
    ]

    gap_summary = pd.DataFrame({
        "metric": [
            "Total interval records",
            "Missing leader gaps",
            "Non-numeric/lap-deficit gaps",
            "Numeric leader gaps",
        ],
        "value": [
            len(intervals),
            intervals["gap_to_leader"].isna().sum(),
            len(lapped_records),
            len(gap_strings) - len(lapped_records),
        ],
    })

    write("7. INTERVAL DATA", gap_summary)

    # ------------------------------------------------
    # SAVE REPORT
    # ------------------------------------------------

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Mapping audit completed successfully!")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
