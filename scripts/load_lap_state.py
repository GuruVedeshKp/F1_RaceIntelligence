"""Build, validate, and optionally load lap_state features for one race.

Default mode is a read-only preflight. Pass --apply to insert records.
Pass --replace together with --apply to replace this race's existing lap_state
records inside the same transaction.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, insert, select, text

from f1ri.db.models import Lap, LapState
from f1ri.db.session import engine
from f1ri.features.lap_state import build_lap_state

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data/raw/openf1/2024/bahrain_9472"
DEFAULT_PROCESSED_DIR = ROOT / "data/processed/bahrain_2024"

LOGGER = logging.getLogger("load_lap_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and load completed-lap features into PostgreSQL."
    )
    parser.add_argument("--race-id", type=int, default=1)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit the feature rows to PostgreSQL. Without this flag, run read-only preflight.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete and rebuild this race's lap_state rows (requires --apply).",
    )
    args = parser.parse_args()
    if args.replace and not args.apply:
        parser.error("--replace requires --apply")
    return args


def read_parquet(folder: Path, name: str) -> pd.DataFrame:
    path = folder / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Required Parquet file does not exist: {path}")
    return pd.read_parquet(path)


def load_race_metadata(race_id: int) -> dict[str, Any]:
    statement = text(
        """
        SELECT race_id, season, round, circuit, date, total_laps,
               source, source_meeting_key, source_session_key
        FROM races
        WHERE race_id = :race_id
        """
    )
    with engine.connect() as connection:
        row = connection.execute(statement, {"race_id": race_id}).mappings().one_or_none()
    if row is None:
        raise ValueError(f"No race exists with race_id={race_id}")
    return dict(row)


def validate_features(features: pd.DataFrame, expected_lap_rows: int) -> dict[str, bool]:
    """Run feature-quality assertions before any database write."""
    required_columns = list(LapState.__table__.columns.keys())
    if list(features.columns) != required_columns:
        missing = sorted(set(required_columns) - set(features.columns))
        extra = sorted(set(features.columns) - set(required_columns))
        raise ValueError(
            "Feature columns do not exactly match LapState. "
            f"Missing={missing}; extra={extra}; "
            f"actual_order={list(features.columns)}"
        )

    checks: dict[str, bool] = {
        "row count matches source laps": len(features) == expected_lap_rows,
        "20 drivers represented": features["driver"].nunique() == 20,
        "no duplicate composite keys": not features.duplicated(
            ["race_id", "driver", "lap"]
        ).any(),
        "positions within 1-20": features["position"].between(1, 20).all(),
        "all grid positions populated": features["grid_position"].notna().all(),
        "grid position constant per driver": features.groupby("driver")[
            "grid_position"
        ].nunique().eq(1).all(),
        "all grid positions represented": set(
            features["grid_position"].dropna().astype(int)
        ) == set(range(1, 21)),
        "no negative leader gaps": features["gap_to_leader_s"].dropna().ge(0).all(),
        "no negative ahead gaps": features["gap_to_ahead_s"].dropna().ge(0).all(),
        "no negative behind gaps": features["gap_to_behind_s"].dropna().ge(0).all(),
        "lapped drivers have no numeric leader gap": features.loc[
            features["laps_behind_leader"].gt(0), "gap_to_leader_s"
        ].isna().all(),
        "leader has no car-ahead gap": features.loc[
            features["position"].eq(1), "gap_to_ahead_s"
        ].isna().all(),
        "laps remaining are nonnegative": features["laps_remaining"].ge(0).all(),
        "grid position does not vary per driver": features.groupby("driver")[
            "grid_position"
        ].nunique().eq(1).all(),
        "pit-stop count never decreases": features.sort_values(
            ["driver", "lap"]
        ).groupby("driver")["stops_made"].apply(
            lambda values: values.is_monotonic_increasing
        ).all(),
        "all prediction targets populated": features[
            ["finish_position", "is_win", "is_top3", "is_top10", "is_dnf"]
        ].notna().all().all(),
    }

    driver_targets = features.drop_duplicates("driver")
    checks["exactly one winner"] = int(driver_targets["is_win"].sum()) == 1
    checks["exactly three podium finishers"] = int(driver_targets["is_top3"].sum()) == 3
    checks["exactly ten top-ten finishers"] = int(driver_targets["is_top10"].sum()) == 10

    print("\nFEATURE QUALITY CHECKS")
    print("=" * 68)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} | {name}")

    print("\nFEATURE SUMMARY")
    print(f"Rows: {len(features):,}")
    print(f"Drivers: {features['driver'].nunique()}")
    print(f"Columns: {len(features.columns)}")
    print(f"Grid coverage: {features['grid_position'].notna().sum():,}/{len(features):,}")
    print(f"Rolling pace coverage: {features['rolling_pace_s'].notna().sum():,}/{len(features):,}")
    print(f"Leader-gap coverage: {features['gap_to_leader_s'].notna().sum():,}/{len(features):,}")
    print(f"Ahead-gap coverage: {features['gap_to_ahead_s'].notna().sum():,}/{len(features):,}")
    print(f"Behind-gap coverage: {features['gap_to_behind_s'].notna().sum():,}/{len(features):,}")
    print(f"Targets: wins={int(driver_targets['is_win'].sum())}, "
          f"podiums={int(driver_targets['is_top3'].sum())}, "
          f"top10={int(driver_targets['is_top10'].sum())}, "
          f"DNFs={int(driver_targets['is_dnf'].sum())}")

    diagnostics = features.attrs.get("diagnostics", {})
    print("\nDATA-QUALITY DIAGNOSTICS")
    for key in [
        "invalid_position_order_snapshots",
        "future_position_matches",
        "future_interval_matches",
        "future_weather_matches",
        "stale_weather_rows",
        "grid_positions_missing_in_source",
    ]:
        if key in diagnostics:
            print(f"{key}: {diagnostics[key]}")

    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Feature validation failed: {failed}")

    # Non-fatal diagnostic: ambiguous race-wide position order is retained,
    # but the feature builder should leave derived behind gaps unavailable.
    invalid_snapshots = int(diagnostics.get("invalid_position_order_snapshots", 0))
    if invalid_snapshots:
        LOGGER.warning(
            "%s reference snapshots have ambiguous position ordering; "
            "the builder masks derived behind-gap values for those snapshots.",
            invalid_snapshots,
        )

    return checks


def to_python_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas/numpy scalars and missing values to DBAPI-safe values."""
    records: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        clean_record: dict[str, Any] = {}
        for key, value in record.items():
            if pd.isna(value):
                clean_record[key] = None
            elif isinstance(value, np.generic):
                clean_record[key] = value.item()
            elif isinstance(value, pd.Timestamp):
                clean_record[key] = value.to_pydatetime()
            else:
                clean_record[key] = value
        records.append(clean_record)
    return records


def validate_database_keys(connection, race_id: int, features: pd.DataFrame) -> int:
    """Require every feature key to correspond to an existing laps row."""
    database_keys = {
        (str(driver), int(lap))
        for driver, lap in connection.execute(
            select(Lap.driver, Lap.lap).where(Lap.race_id == race_id)
        ).all()
    }
    feature_keys = set(
        zip(features["driver"].astype(str), features["lap"].astype(int))
    )
    missing_from_features = database_keys - feature_keys
    missing_from_laps = feature_keys - database_keys
    if missing_from_features or missing_from_laps:
        raise ValueError(
            "Feature/laps key mismatch. "
            f"Missing feature keys (first 10): {sorted(missing_from_features)[:10]}; "
            f"features without lap rows (first 10): {sorted(missing_from_laps)[:10]}"
        )
    return len(database_keys)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    race = load_race_metadata(args.race_id)
    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()

    LOGGER.info(
        "Building features for race_id=%s, %s round %s (%s)",
        args.race_id,
        race["season"],
        race["round"],
        race["circuit"],
    )

    features = build_lap_state(
        race_id=args.race_id,
        race=race,
        laps=read_parquet(processed_dir, "laps"),
        positions=read_parquet(raw_dir, "position"),
        intervals=read_parquet(raw_dir, "intervals"),
        weather=read_parquet(raw_dir, "weather"),
        pit_stops=read_parquet(processed_dir, "pit_stops"),
        results=read_parquet(processed_dir, "results"),
    )

    # Confirm the feature rows map one-to-one to persisted completed laps.
    with engine.connect() as connection:
        expected_lap_rows = int(
            connection.execute(
                select(func.count()).select_from(Lap).where(Lap.race_id == args.race_id)
            ).scalar_one()
        )
        validate_database_keys(connection, args.race_id, features)

    validate_features(features, expected_lap_rows)
    records = to_python_records(features)

    print("\nDATABASE PREFLIGHT")
    print("=" * 68)
    print(f"Race ID: {args.race_id}")
    print(f"Existing completed-lap rows: {expected_lap_rows:,}")
    print(f"Feature records prepared: {len(records):,}")
    print(f"Existing lap_state rows will be {'replaced' if args.replace else 'checked'}.")

    if not args.apply:
        with engine.connect() as connection:
            existing = int(
                connection.execute(
                    select(func.count()).select_from(LapState).where(
                        LapState.race_id == args.race_id
                    )
                ).scalar_one()
            )
        if existing:
            raise ValueError(
                f"Found {existing} existing lap_state rows for race_id={args.race_id}. "
                "Use --apply --replace only if you intend to rebuild them."
            )
        print("\nREAD-ONLY PREFLIGHT PASSED. No database rows were written.")
        print("To insert these records, run this command explicitly:")
        print(
            f"uv run python scripts/load_lap_state.py --race-id {args.race_id} --apply"
        )
        return

    # One transaction covers optional replacement, insert, and post-insert checks.
    with engine.begin() as connection:
        validate_database_keys(connection, args.race_id, features)
        existing = int(
            connection.execute(
                select(func.count()).select_from(LapState).where(
                    LapState.race_id == args.race_id
                )
            ).scalar_one()
        )

        if existing and not args.replace:
            raise ValueError(
                f"Found {existing} existing lap_state rows for race_id={args.race_id}. "
                "Refusing to insert duplicates. Use --replace to rebuild this race."
            )

        if existing and args.replace:
            connection.execute(
                delete(LapState).where(LapState.race_id == args.race_id)
            )
            LOGGER.warning("Deleted %s existing lap_state rows for replacement.", existing)

        connection.execute(insert(LapState.__table__), records)

        inserted = int(
            connection.execute(
                select(func.count()).select_from(LapState).where(
                    LapState.race_id == args.race_id
                )
            ).scalar_one()
        )
        if inserted != len(records):
            raise RuntimeError(
                f"Post-insert verification failed: expected {len(records)}, found {inserted}"
            )

    print("\nIMPORT COMMITTED SUCCESSFULLY")
    print(f"Race ID: {args.race_id}")
    print(f"lap_state rows: {inserted:,}")
    print("All feature records match existing laps composite keys.")


if __name__ == "__main__":
    main()
