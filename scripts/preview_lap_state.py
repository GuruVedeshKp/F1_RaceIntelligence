"""Build and validate Bahrain lap-state features in memory; no DB writes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from f1ri.db.session import engine
from f1ri.features.lap_state import build_lap_state


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview leak-safe lap-state features without inserting them."
    )
    parser.add_argument("--race-id", type=int, default=1)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/openf1/2024/bahrain_9472"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/bahrain_2024"),
    )
    args = parser.parse_args()

    with engine.connect() as connection:
        race_row = connection.execute(
            text(
                """
                SELECT race_id, season, round, circuit, date, total_laps,
                       source, source_meeting_key, source_session_key
                FROM races
                WHERE race_id = :race_id
                """
            ),
            {"race_id": args.race_id},
        ).mappings().one_or_none()

    if race_row is None:
        raise SystemExit(f"No race found with race_id={args.race_id}")

    def read_parquet(folder: Path, name: str) -> pd.DataFrame:
        path = folder / f"{name}.parquet"
        if not path.exists():
            raise SystemExit(f"Required Parquet file does not exist: {path}")
        return pd.read_parquet(path)

    features = build_lap_state(
        race_id=args.race_id,
        race=dict(race_row),
        laps=read_parquet(args.processed_dir, "laps"),
        positions=read_parquet(args.raw_dir, "position"),
        intervals=read_parquet(args.raw_dir, "intervals"),
        weather=read_parquet(args.raw_dir, "weather"),
        pit_stops=read_parquet(args.processed_dir, "pit_stops"),
        results=read_parquet(args.processed_dir, "results"),
    )

    print("\nLAP_STATE PREVIEW — NO DATABASE WRITES")
    print("=" * 68)
    print(f"Race: {race_row['season']} round {race_row['round']} — {race_row['circuit']}")
    print(f"Rows generated: {len(features):,}")
    print(f"Drivers: {features['driver'].nunique()}")
    print(f"Lap range: {features['lap'].min()}–{features['lap'].max()}")
    print(f"Columns: {len(features.columns)}")

    print("\nFEATURE COVERAGE")
    for column in [
        "position", "gap_to_leader_s", "laps_behind_leader",
        "gap_to_ahead_s", "gap_to_behind_s", "compound", "tyre_age",
        "stint", "stops_made", "grid_position", "rolling_pace_s",
        "pace_delta_to_leader_s", "air_temp", "track_temp", "rain",
    ]:
        count = int(features[column].notna().sum())
        print(f"{column:28s} {count:4d}/{len(features):4d} non-null")

    print("\nTARGET COVERAGE")
    for column in ["finish_position", "is_win", "is_top3", "is_top10", "is_dnf"]:
        print(f"{column:28s} {features[column].notna().sum():4d}/{len(features):4d} non-null")

    print("\nDIAGNOSTICS")
    for key, value in features.attrs.get("diagnostics", {}).items():
        print(f"{key}: {value}")

    print("\nSAMPLE FEATURES")
    sample_columns = [
        "driver", "lap", "position", "gap_to_leader_s", "laps_behind_leader",
        "gap_to_ahead_s", "gap_to_behind_s", "compound", "tyre_age",
        "stops_made", "rolling_pace_s", "pace_delta_to_leader_s",
        "air_temp", "track_temp", "finish_position", "is_dnf",
    ]
    print(features[sample_columns].head(20).to_string(index=False))

    # Semantic guard: OpenF1 reports 0.0 for the leader's interval, but a
    # leader has no car ahead. The feature builder must represent that as NULL.
    leader_rows = features.loc[features["position"] == 1]
    if leader_rows["gap_to_ahead_s"].notna().any():
        raise SystemExit("Validation failed: leader rows have a gap_to_ahead_s value")

    print("\nLeader gap-to-ahead semantic check: passed")
    print("Validation complete. Nothing was inserted into PostgreSQL.")


if __name__ == "__main__":
    main()
