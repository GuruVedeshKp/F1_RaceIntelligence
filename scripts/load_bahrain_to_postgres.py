"""Load the processed 2024 Bahrain GP OpenF1 datasets into PostgreSQL.

Run from the project root:
    uv run python scripts/load_bahrain_to_postgres.py --check-only
    uv run python scripts/load_bahrain_to_postgres.py

To deliberately replace an existing import for this race:
    uv run python scripts/load_bahrain_to_postgres.py --replace
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, func, insert, or_, select

from f1ri.db.models import Lap, PitStop, Race, RaceResult, TrackEvent, Weather
from f1ri.db.session import SessionLocal

# This loader is intentionally specific to the verified Bahrain 2024 dataset.
SEASON = 2024
ROUND = 1
CIRCUIT = "Bahrain International Circuit"
RACE_DATE = date(2024, 3, 2)
TOTAL_LAPS = 57
SOURCE = "openf1"
MEETING_KEY = 1229
SESSION_KEY = 9472

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bahrain_2024"

EXPECTED_ROWS = {
    "laps": 1129,
    "pit_stops": 43,
    "weather": 57,
    "track_events": 9,
    "results": 20,
}

REQUIRED_COLUMNS = {
    "laps": {
        "meeting_key", "session_key", "driver_number", "driver", "lap",
        "lap_start_utc", "lap_end_utc", "lap_time_s", "position",
        "compound", "tyre_age", "stint", "pit_in", "pit_out",
        "track_status", "is_clean", "position_state_age_s",
    },
    "pit_stops": {
        "meeting_key", "session_key", "driver_number", "driver", "lap",
        "pit_time_utc", "lane_duration_s", "stationary_duration_s",
        "pit_loss_s", "under_sc",
    },
    "weather": {"lap", "air_temp", "track_temp", "rain", "weather_age_s"},
    "track_events": {
        "meeting_key", "session_key", "type", "start_lap", "end_lap",
        "event_time_utc", "category", "flag", "scope", "sector",
        "driver_number", "message",
    },
    "results": {
        "meeting_key", "session_key", "driver_number", "driver", "team",
        "grid", "finish_pos", "number_of_laps", "status", "points",
        "duration_s", "gap_to_leader_raw", "dnf", "dns", "dsq",
    },
}

# These NULLs were observed in the data-quality audit and are intentionally
# preserved. Any newly missing value outside this allowlist stops the import.
ALLOWED_NULLS = {
    "laps": {"track_status"},
    "pit_stops": {"stationary_duration_s", "pit_loss_s"},
    "weather": set(),
    "track_events": {"end_lap", "sector", "driver_number"},
    "results": {"grid", "duration_s"},
}

UNIQUE_KEYS = {
    "laps": ["driver", "lap"],
    "pit_stops": ["driver", "lap"],
    "weather": ["lap"],
    "results": ["driver"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import the processed 2024 Bahrain GP datasets into PostgreSQL."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate Parquet files and check for an existing race without writing.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Delete an existing matching race and its dependent rows before re-importing. "
            "Use only when you intentionally want to replace this race's database data."
        ),
    )
    return parser.parse_args()


def load_and_validate_files() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Processed-data directory not found: {DATA_DIR}")

    for name, expected_count in EXPECTED_ROWS.items():
        path = DATA_DIR / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Required Parquet file not found: {path}")

        frame = pd.read_parquet(path)
        frames[name] = frame

        if len(frame) != expected_count:
            raise ValueError(
                f"{name}.parquet has {len(frame)} rows; expected {expected_count}. "
                "This loader is pinned to the audited Bahrain 2024 dataset."
            )

        missing_columns = REQUIRED_COLUMNS[name] - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{name}.parquet is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        unexpected_nulls = {
            column: int(frame[column].isna().sum())
            for column in frame.columns
            if frame[column].isna().any() and column not in ALLOWED_NULLS[name]
        }
        if unexpected_nulls:
            raise ValueError(
                f"{name}.parquet has unexpected missing values: {unexpected_nulls}"
            )

        # The source identifiers must match the one session this loader imports.
        if "meeting_key" in frame.columns:
            meeting_keys = set(frame["meeting_key"].dropna().astype(int).unique())
            session_keys = set(frame["session_key"].dropna().astype(int).unique())
            if meeting_keys != {MEETING_KEY} or session_keys != {SESSION_KEY}:
                raise ValueError(
                    f"{name}.parquet contains unexpected OpenF1 identifiers: "
                    f"meeting_key={meeting_keys}, session_key={session_keys}"
                )

        if name in UNIQUE_KEYS:
            key_columns = UNIQUE_KEYS[name]
            if frame.duplicated(subset=key_columns).any():
                raise ValueError(
                    f"{name}.parquet contains duplicate keys: {key_columns}"
                )

    # Validate lap numbers against the known 57-lap race.
    if not frames["laps"]["lap"].between(1, TOTAL_LAPS).all():
        raise ValueError("laps.parquet contains lap numbers outside 1..57")
    if not frames["weather"]["lap"].between(1, TOTAL_LAPS).all():
        raise ValueError("weather.parquet contains lap numbers outside 1..57")

    # Driver codes are stored in VARCHAR(3) columns in the database.
    for dataset in ("laps", "pit_stops", "results"):
        too_long = frames[dataset]["driver"].astype(str).str.len() > 3
        if too_long.any():
            bad_values = frames[dataset].loc[too_long, "driver"].unique().tolist()
            raise ValueError(
                f"{dataset}.parquet has driver codes longer than 3 characters: "
                f"{bad_values}"
            )

    return frames


def to_python_value(value: Any) -> Any:
    """Convert pandas/numpy scalars and missing values to DB-friendly Python values."""
    if value is None or pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"Naive datetime encountered ({value!r}); expected a timezone-aware UTC value."
            )
        return value.astimezone(UTC)

    # Convert numpy scalar types (int64, float64, bool_, etc.) to native Python.
    if hasattr(value, "item") and callable(value.item):
        return value.item()

    return value


def make_insert_rows(
    frame: pd.DataFrame,
    column_mapping: dict[str, str],
    race_id: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_row in frame.to_dict(orient="records"):
        row: dict[str, Any] = {"race_id": race_id}
        for source_column, model_attribute in column_mapping.items():
            row[model_attribute] = to_python_value(source_row[source_column])
        rows.append(row)
    return rows


def find_existing_races(session) -> list[Race]:
    statement = select(Race).where(
        or_(
            (Race.source == SOURCE) & (Race.source_session_key == SESSION_KEY),
            (Race.season == SEASON) & (Race.round == ROUND),
        )
    )
    return list(session.scalars(statement).all())


def count_rows(session, model, race_id: int) -> int:
    return int(
        session.scalar(
            select(func.count()).select_from(model).where(model.race_id == race_id)
        )
        or 0
    )


def main() -> int:
    args = parse_args()

    if args.check_only and args.replace:
        raise ValueError("Use either --check-only or --replace, not both.")

    frames = load_and_validate_files()
    print("Parquet validation passed:")
    for name, frame in frames.items():
        print(f"  {name}: {len(frame):,} rows")

    with SessionLocal() as session:
        if args.check_only:
            existing = find_existing_races(session)
            if existing:
                print("\nA matching 2024 Bahrain race already exists:")
                for race in existing:
                    print(
                        f"  race_id={race.race_id}, season={race.season}, "
                        f"round={race.round}, source={race.source}, "
                        f"source_session_key={race.source_session_key}"
                    )
                print("No database changes were made.")
            else:
                print("\nNo matching race exists. The database is ready for import.")
            print("Check-only complete; no database changes were made.")
            return 0

        with session.begin():
            existing = find_existing_races(session)
            if existing and not args.replace:
                details = ", ".join(
                    f"race_id={race.race_id} (season={race.season}, round={race.round}, "
                    f"session_key={race.source_session_key})"
                    for race in existing
                )
                raise RuntimeError(
                    "A matching race already exists: "
                    f"{details}. No rows were changed. Use --replace only if you "
                    "intentionally want to delete and reload this race."
                )

            if existing and args.replace:
                existing_ids = [race.race_id for race in existing]
                session.execute(
                    delete(Race)
                    .where(Race.race_id.in_(existing_ids))
                    .execution_options(synchronize_session=False)
                )
                session.flush()
                print(
                    "Removed existing matching race record(s); PostgreSQL foreign-key "
                    "CASCADE will remove their dependent records in this transaction."
                )

            race = Race(
                season=SEASON,
                round=ROUND,
                circuit=CIRCUIT,
                date=RACE_DATE,
                total_laps=TOTAL_LAPS,
                source=SOURCE,
                source_meeting_key=MEETING_KEY,
                source_session_key=SESSION_KEY,
                # Session start/end are intentionally left NULL until sourced directly
                # from the saved OpenF1 session metadata.
            )
            session.add(race)
            session.flush()
            race_id = race.race_id

            lap_rows = make_insert_rows(
                frames["laps"],
                {
                    "driver": "driver",
                    "lap": "lap",
                    "lap_time_s": "lap_time_s",
                    "position": "position",
                    "compound": "compound",
                    "tyre_age": "tyre_age",
                    "stint": "stint",
                    "pit_in": "pit_in",
                    "pit_out": "pit_out",
                    "track_status": "track_status",
                    "is_clean": "is_clean",
                    "driver_number": "driver_number",
                    "lap_start_utc": "lap_start_utc",
                    "lap_end_utc": "lap_end_utc",
                    "position_state_age_s": "position_state_age_s",
                },
                race_id,
            )
            result_rows = make_insert_rows(
                frames["results"],
                {
                    "driver": "driver",
                    "team": "team",
                    "grid": "grid",
                    "finish_pos": "finish_pos",
                    "status": "status",
                    "points": "points",
                    "driver_number": "driver_number",
                    "number_of_laps": "number_of_laps",
                    "duration_s": "duration_s",
                    "gap_to_leader_raw": "gap_to_leader_raw",
                    "dnf": "dnf",
                    "dns": "dns",
                    "dsq": "dsq",
                },
                race_id,
            )
            weather_rows = make_insert_rows(
                frames["weather"],
                {
                    "lap": "lap",
                    "air_temp": "air_temp",
                    "track_temp": "track_temp",
                    "rain": "rain",
                    "weather_age_s": "weather_age_s",
                },
                race_id,
            )
            pit_stop_rows = make_insert_rows(
                frames["pit_stops"],
                {
                    "driver": "driver",
                    "lap": "lap",
                    "pit_loss_s": "pit_loss_s",
                    "under_sc": "under_sc",
                    "driver_number": "driver_number",
                    "pit_time_utc": "pit_time_utc",
                    "lane_duration_s": "lane_duration_s",
                    "stationary_duration_s": "stationary_duration_s",
                },
                race_id,
            )
            track_event_rows = make_insert_rows(
                frames["track_events"],
                {
                    # TrackEvent.event_type is mapped to the SQL column named "type".
                    "type": "event_type",
                    "start_lap": "start_lap",
                    "end_lap": "end_lap",
                    "message": "message",
                    "event_time_utc": "event_time_utc",
                    "category": "category",
                    "flag": "flag",
                    "scope": "scope",
                    "sector": "sector",
                    "driver_number": "driver_number",
                },
                race_id,
            )

            # Insert the race first, then all rows with its generated race_id.
            session.execute(insert(Lap), lap_rows)
            session.execute(insert(RaceResult), result_rows)
            session.execute(insert(Weather), weather_rows)
            session.execute(insert(PitStop), pit_stop_rows)
            session.execute(insert(TrackEvent), track_event_rows)

            inserted_counts = {
                "laps": count_rows(session, Lap, race_id),
                "results": count_rows(session, RaceResult, race_id),
                "weather": count_rows(session, Weather, race_id),
                "pit_stops": count_rows(session, PitStop, race_id),
                "track_events": count_rows(session, TrackEvent, race_id),
            }
            expected_by_table = {
                "laps": EXPECTED_ROWS["laps"],
                "results": EXPECTED_ROWS["results"],
                "weather": EXPECTED_ROWS["weather"],
                "pit_stops": EXPECTED_ROWS["pit_stops"],
                "track_events": EXPECTED_ROWS["track_events"],
            }
            if inserted_counts != expected_by_table:
                raise RuntimeError(
                    f"Inserted row-count verification failed: {inserted_counts}; "
                    f"expected {expected_by_table}. Transaction will be rolled back."
                )

        print("\nImport committed successfully.")
        print(f"  race_id: {race_id}")
        print(f"  season / round: {SEASON} / {ROUND}")
        print(f"  source session: {SESSION_KEY} (meeting {MEETING_KEY})")
        print("  verified database row counts:")
        for table, row_count in inserted_counts.items():
            print(f"    {table}: {row_count:,}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise