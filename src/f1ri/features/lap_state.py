"""Build leak-safe, completed-lap features for F1 Race Intelligence.

This module only constructs an in-memory DataFrame. It does not write to the
PostgreSQL database. The companion preview script loads the source Parquet
files and prints validation diagnostics before any database insert is added.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

INTERVAL_MAX_AGE_S = 5.0
WEATHER_MAX_AGE_S = 60.0
ROLLING_CLEAN_LAPS = 3


class FeatureBuildError(ValueError):
    """Raised when source data cannot safely produce lap-state features."""


def _require_columns(frame: pd.DataFrame, name: str, required: set[str]) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise FeatureBuildError(f"{name} is missing required columns: {missing}")


def _parse_utc(values: pd.Series, label: str) -> pd.Series:
    """Parse mixed ISO-8601 strings (with/without fractional seconds) as UTC."""
    try:
        return pd.to_datetime(values, format="ISO8601", utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise FeatureBuildError(f"Could not parse {label} as ISO-8601 UTC timestamps") from exc


def _clean_boolean(values: pd.Series, label: str) -> pd.Series:
    """Normalize a source boolean column without treating missing as False."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean")
    mapping = {
        True: True,
        False: False,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }
    result = values.map(mapping).astype("boolean")
    bad = values.notna() & result.isna()
    if bad.any():
        examples = values[bad].astype(str).value_counts().head(5).to_dict()
        raise FeatureBuildError(f"Unexpected values in {label}: {examples}")
    return result


def _build_position_interval_snapshots(
    lap_rows: pd.DataFrame,
    positions: pd.DataFrame,
    intervals: pd.DataFrame,
    interval_max_age_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct the 20-car state at every completed-lap timestamp.

    All joins are backward as-of joins. Position is event-driven and is carried
    forward; interval-derived quantities are accepted only within the configured
    freshness limit. The returned own_state has one row per lap_rows row.
    """
    driver_numbers = sorted(lap_rows["driver_number"].astype(int).unique().tolist())
    target_times = lap_rows[["row_id", "lap_end_utc"]].rename(
        columns={"lap_end_utc": "snapshot_utc"}
    )
    target_times["_cross_key"] = 1
    driver_frame = pd.DataFrame({"state_driver_number": driver_numbers})
    driver_frame["_cross_key"] = 1
    snapshot_grid = target_times.merge(driver_frame, on="_cross_key", how="inner").drop(
        columns="_cross_key"
    )
    snapshot_grid = snapshot_grid.sort_values("snapshot_utc", kind="mergesort")

    # Event-driven positions: use the latest position event at or before snapshot.
    pos = positions[["driver_number", "date", "position"]].copy()
    pos["driver_number"] = pd.to_numeric(pos["driver_number"], errors="raise").astype(int)
    pos["position"] = pd.to_numeric(pos["position"], errors="coerce")
    pos = pos.rename(
        columns={
            "driver_number": "state_driver_number",
            "date": "position_event_utc",
            "position": "state_position",
        }
    )
    pos = pos.dropna(subset=["state_position", "position_event_utc"])
    pos = pos.sort_values("position_event_utc", kind="mergesort")
    pos = pos.drop_duplicates(
        ["state_driver_number", "position_event_utc"], keep="last"
    )

    position_state = pd.merge_asof(
        snapshot_grid,
        pos,
        left_on="snapshot_utc",
        right_on="position_event_utc",
        by="state_driver_number",
        direction="backward",
        allow_exact_matches=True,
    )
    position_state["position_age_s"] = (
        position_state["snapshot_utc"] - position_state["position_event_utc"]
    ).dt.total_seconds()

    # Intervals: numeric gaps and lap deficits are separate features.
    ints = intervals[
        ["driver_number", "date", "gap_to_leader", "interval"]
    ].copy()
    ints["driver_number"] = pd.to_numeric(ints["driver_number"], errors="raise").astype(int)
    raw_gap = ints["gap_to_leader"].astype("string").str.strip()
    ints["leader_gap_s"] = pd.to_numeric(raw_gap, errors="coerce")
    ints["interval_s"] = pd.to_numeric(ints["interval"], errors="coerce")

    lap_deficit_text = raw_gap.str.extract(
        r"^\+(\d+)\s+LAPS?$", flags=re.IGNORECASE, expand=False
    )
    ints["laps_behind_leader"] = pd.Series(
        pd.NA, index=ints.index, dtype="Int64"
    )
    ints.loc[ints["leader_gap_s"].notna(), "laps_behind_leader"] = 0
    lap_deficits = pd.to_numeric(lap_deficit_text, errors="coerce").astype("Int64")
    ints.loc[lap_deficits.notna(), "laps_behind_leader"] = lap_deficits[lap_deficits.notna()]

    recognized = ints["leader_gap_s"].notna() | ints["laps_behind_leader"].notna()
    unknown_gap_values = raw_gap.notna() & ~recognized
    unknown_gap_examples = (
        raw_gap[unknown_gap_values].value_counts().head(10).to_dict()
        if unknown_gap_values.any()
        else {}
    )

    ints = ints.rename(
        columns={
            "driver_number": "state_driver_number",
            "date": "interval_event_utc",
        }
    )
    ints = ints.sort_values("interval_event_utc", kind="mergesort")
    ints = ints.drop_duplicates(
        ["state_driver_number", "interval_event_utc"], keep="last"
    )

    interval_state = pd.merge_asof(
        snapshot_grid,
        ints[
            [
                "state_driver_number",
                "interval_event_utc",
                "leader_gap_s",
                "laps_behind_leader",
                "interval_s",
            ]
        ],
        left_on="snapshot_utc",
        right_on="interval_event_utc",
        by="state_driver_number",
        direction="backward",
        allow_exact_matches=True,
    )
    interval_state["interval_age_s"] = (
        interval_state["snapshot_utc"] - interval_state["interval_event_utc"]
    ).dt.total_seconds()
    future_interval_matches = int(
        (interval_state["interval_event_utc"] > interval_state["snapshot_utc"]).sum()
    )
    if future_interval_matches:
        raise FeatureBuildError("Interval as-of join unexpectedly used future observations")

    interval_is_fresh = interval_state["interval_age_s"].between(
        0, interval_max_age_s, inclusive="both"
    )
    for column in ["leader_gap_s", "laps_behind_leader", "interval_s"]:
        interval_state.loc[~interval_is_fresh, column] = pd.NA

    state = position_state.merge(
        interval_state[
            [
                "row_id",
                "state_driver_number",
                "interval_event_utc",
                "leader_gap_s",
                "laps_behind_leader",
                "interval_s",
                "interval_age_s",
            ]
        ],
        on=["row_id", "state_driver_number"],
        how="left",
        validate="one_to_one",
    )

    # A complete ordering is needed to infer the car immediately behind.
    # If an event-time snapshot temporarily contains duplicate/missing places,
    # keep position and own-ahead interval but leave behind-gap unavailable.
    snapshot_quality = state.groupby("row_id").agg(
        state_rows=("state_driver_number", "count"),
        positions_present=("state_position", "count"),
        unique_positions=("state_position", "nunique"),
        min_position=("state_position", "min"),
        max_position=("state_position", "max"),
    )
    snapshot_quality["valid_order"] = (
        (snapshot_quality["state_rows"] == len(driver_numbers))
        & (snapshot_quality["positions_present"] == len(driver_numbers))
        & (snapshot_quality["unique_positions"] == len(driver_numbers))
        & (snapshot_quality["min_position"] == 1)
        & (snapshot_quality["max_position"] == len(driver_numbers))
    )
    valid_by_row = snapshot_quality["valid_order"].to_dict()

    state = state.sort_values(
        ["row_id", "state_position", "state_driver_number"], kind="mergesort"
    ).copy()
    state["gap_to_behind_s"] = state.groupby("row_id", sort=False)["interval_s"].shift(-1)
    state["snapshot_valid_order"] = state["row_id"].map(valid_by_row).fillna(False)
    state.loc[~state["snapshot_valid_order"], "gap_to_behind_s"] = np.nan

    # Identify the leader only when the whole snapshot ordering is unambiguous.
    leader_candidates = state.loc[
        state["snapshot_valid_order"] & (state["state_position"] == 1),
        ["row_id", "state_driver_number"],
    ]
    leader_by_row = leader_candidates.drop_duplicates("row_id").set_index("row_id")[
        "state_driver_number"
    ]

    own_keys = lap_rows[["row_id", "driver_number"]].copy()
    own_state = own_keys.merge(
        state,
        left_on=["row_id", "driver_number"],
        right_on=["row_id", "state_driver_number"],
        how="left",
        validate="one_to_one",
    )
    if own_state["state_position"].isna().any():
        missing = int(own_state["state_position"].isna().sum())
        raise FeatureBuildError(
            f"Could not reconstruct position for {missing} target driver-lap rows"
        )

    own_state = own_state[
        [
            "row_id",
            "state_position",
            "position_event_utc",
            "position_age_s",
            "leader_gap_s",
            "laps_behind_leader",
            "interval_s",
            "gap_to_behind_s",
            "interval_age_s",
            "snapshot_valid_order",
        ]
    ].rename(
        columns={
            "state_position": "position",
            "leader_gap_s": "gap_to_leader_s",
            "interval_s": "gap_to_ahead_s",
        }
    )
    # OpenF1 reports the leader's interval as 0.0. That is not a real gap
    # to a car ahead because the leader has no car ahead of it. Keep the raw
    # interval for behind-gap reconstruction above, but null this feature.
    own_state.loc[own_state["position"] == 1, "gap_to_ahead_s"] = np.nan
    own_state["leader_driver_number"] = own_state["row_id"].map(leader_by_row)

    diagnostics = {
        "snapshot_count": len(lap_rows),
        "drivers_in_snapshot": len(driver_numbers),
        "invalid_position_order_snapshots": int((~snapshot_quality["valid_order"]).sum()),
        "max_position_age_s": float(position_state["position_age_s"].max()),
        "max_interval_age_s": float(interval_state["interval_age_s"].max()),
        "stale_interval_state_rows": int((~interval_is_fresh).sum()),
        "unknown_non_numeric_leader_gap_values": unknown_gap_examples,
        "future_position_matches": int(
            (position_state["position_event_utc"] > position_state["snapshot_utc"]).sum()
        ),
        "future_interval_matches": future_interval_matches,
    }
    return own_state, state, diagnostics


def _add_rolling_pace(lap_rows: pd.DataFrame, window: int) -> pd.DataFrame:
    """Use the latest `window` clean completed laps, including current lap if clean."""
    frame = lap_rows.copy()
    frame["rolling_pace_s"] = np.nan

    for _, group in frame.groupby("driver_number", sort=False):
        clean = group.loc[group["is_clean"].fillna(False)].sort_values(
            ["lap_end_utc", "lap"], kind="mergesort"
        )
        clean = clean.dropna(subset=["lap_time_s"])
        rolling = clean["lap_time_s"].rolling(window=window, min_periods=window).mean()
        pace_by_row = pd.Series(rolling.to_numpy(), index=clean["row_id"])
        mask = frame["row_id"].isin(group["row_id"])
        frame.loc[mask, "rolling_pace_s"] = frame.loc[mask, "row_id"].map(pace_by_row)

    frame["rolling_pace_s"] = frame.groupby("driver_number", sort=False)[
        "rolling_pace_s"
    ].ffill()
    return frame


def build_lap_state(
    *,
    race_id: int,
    race: Mapping[str, Any],
    laps: pd.DataFrame,
    positions: pd.DataFrame,
    intervals: pd.DataFrame,
    weather: pd.DataFrame,
    pit_stops: pd.DataFrame,
    results: pd.DataFrame,
    interval_max_age_s: float = INTERVAL_MAX_AGE_S,
    weather_max_age_s: float = WEATHER_MAX_AGE_S,
    rolling_clean_laps: int = ROLLING_CLEAN_LAPS,
) -> pd.DataFrame:
    """Build one feature row per actual completed driver-lap.

    Parameters are in-memory source frames. `race` must contain `season`,
    `circuit`, and `total_laps`. The returned columns match the LapState ORM
    model. No database writes are performed.
    """
    if rolling_clean_laps < 1:
        raise FeatureBuildError("rolling_clean_laps must be at least 1")
    if interval_max_age_s <= 0 or weather_max_age_s <= 0:
        raise FeatureBuildError("Freshness limits must be positive")

    _require_columns(
        laps,
        "laps",
        {
            "driver_number", "driver", "lap", "lap_end_utc", "lap_time_s",
            "position", "compound", "tyre_age", "stint", "pit_in", "pit_out",
            "track_status", "is_clean",
        },
    )
    _require_columns(positions, "positions", {"driver_number", "date", "position"})
    _require_columns(
        intervals, "intervals", {"driver_number", "date", "gap_to_leader", "interval"}
    )
    _require_columns(
        weather,
        "weather",
        {"date", "air_temperature", "track_temperature", "rainfall"},
    )
    _require_columns(pit_stops, "pit_stops", {"driver_number", "pit_time_utc"})
    _require_columns(
        results,
        "results",
        {"driver", "team", "grid", "finish_pos", "dnf"},
    )

    if "season" not in race or "circuit" not in race or "total_laps" not in race:
        raise FeatureBuildError("race must contain season, circuit and total_laps")
    season = int(race["season"])
    total_laps_value = race["total_laps"]
    if pd.isna(total_laps_value):
        raise FeatureBuildError("Race total_laps is NULL; cannot calculate laps_remaining")
    total_laps = int(total_laps_value)
    if total_laps <= 0:
        raise FeatureBuildError("Race total_laps must be positive")

    base = laps.copy().reset_index(drop=True)
    base["processed_position"] = pd.to_numeric(base["position"], errors="coerce")
    base = base.drop(columns=["position"])
    base["row_id"] = np.arange(len(base), dtype=np.int64)
    base["driver_number"] = pd.to_numeric(base["driver_number"], errors="raise").astype(int)
    base["lap"] = pd.to_numeric(base["lap"], errors="raise").astype(int)
    base["lap_end_utc"] = _parse_utc(base["lap_end_utc"], "laps.lap_end_utc")
    base["lap_time_s"] = pd.to_numeric(base["lap_time_s"], errors="coerce")
    base["tyre_age"] = pd.to_numeric(base["tyre_age"], errors="coerce")
    base["stint"] = pd.to_numeric(base["stint"], errors="coerce").astype("Int64")
    base["is_clean"] = _clean_boolean(base["is_clean"], "laps.is_clean")

    duplicate_laps = base.duplicated(["driver", "lap"], keep=False)
    if duplicate_laps.any():
        example = base.loc[duplicate_laps, ["driver", "lap"]].head(10).to_dict("records")
        raise FeatureBuildError(f"Duplicate driver-lap rows found: {example}")
    if base.empty:
        raise FeatureBuildError("No lap rows supplied")
    if (base["lap"] > total_laps).any():
        raise FeatureBuildError("A driver-lap exceeds race.total_laps")

    base = base.sort_values(
        ["driver_number", "lap_end_utc", "lap"], kind="mergesort"
    ).reset_index(drop=True)
    # Restore a stable unique row key after sorting.
    base["row_id"] = np.arange(len(base), dtype=np.int64)

    base = _add_rolling_pace(base, rolling_clean_laps)

    pos = positions.copy()
    pos["date"] = _parse_utc(pos["date"], "positions.date")
    ints = intervals.copy()
    ints["date"] = _parse_utc(ints["date"], "intervals.date")

    own_state, _snapshot_state, position_diagnostics = _build_position_interval_snapshots(
        base, pos, ints, interval_max_age_s
    )
    base = base.merge(own_state, on="row_id", how="left", validate="one_to_one")

    # Confirm the event-time reconstructed position agrees with the previously
    # audited position column in the transformed lap table.
    position_mismatch = (
        base["processed_position"].notna()
        & base["position"].notna()
        & (base["processed_position"] != base["position"])
    )
    if position_mismatch.any():
        sample = base.loc[
            position_mismatch, ["driver", "lap", "processed_position", "position"]
        ].head(10).to_dict("records")
        raise FeatureBuildError(
            f"Position reconstruction differs from transformed laps: {sample}"
        )
    base = base.drop(columns=["processed_position"])

    # Weather is sampled backwards at each individual lap completion timestamp.
    wx = weather[["date", "air_temperature", "track_temperature", "rainfall"]].copy()
    wx["date"] = _parse_utc(wx["date"], "weather.date")
    wx = wx.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last")
    weather_match = pd.merge_asof(
        base.sort_values("lap_end_utc", kind="mergesort"),
        wx,
        left_on="lap_end_utc",
        right_on="date",
        direction="backward",
        allow_exact_matches=True,
    )
    weather_match["weather_age_s"] = (
        weather_match["lap_end_utc"] - weather_match["date"]
    ).dt.total_seconds()
    weather_fresh = weather_match["weather_age_s"].between(
        0, weather_max_age_s, inclusive="both"
    )
    for column in ["air_temperature", "track_temperature", "rainfall"]:
        weather_match.loc[~weather_fresh, column] = np.nan

    base = weather_match.sort_values("row_id", kind="mergesort").reset_index(drop=True)
    base["air_temp"] = pd.to_numeric(base.pop("air_temperature"), errors="coerce")
    base["track_temp"] = pd.to_numeric(base.pop("track_temperature"), errors="coerce")
    rainfall = pd.to_numeric(base.pop("rainfall"), errors="coerce")
    base["rain"] = pd.Series(
        pd.array(rainfall.map(lambda value: pd.NA if pd.isna(value) else bool(value)), dtype="boolean"),
        index=base.index,
    )

    # Count pit stops using the actual pit event timestamp, not just lap number.
    pits = pit_stops[["driver_number", "pit_time_utc"]].copy()
    pits["driver_number"] = pd.to_numeric(pits["driver_number"], errors="raise").astype(int)
    pits["pit_time_utc"] = _parse_utc(pits["pit_time_utc"], "pit_stops.pit_time_utc")
    pits = pits.dropna(subset=["pit_time_utc"]).sort_values("pit_time_utc")
    base["stops_made"] = 0
    for driver_number, indices in base.groupby("driver_number", sort=False).groups.items():
        driver_pit_times = pits.loc[
            pits["driver_number"] == driver_number, "pit_time_utc"
        ].sort_values().array.asi8
        lap_end_ns = base.loc[indices, "lap_end_utc"].array.asi8
        base.loc[indices, "stops_made"] = np.searchsorted(
            driver_pit_times, lap_end_ns, side="right"
        )

    # Join final results only as target labels, never as input features.
    result_cols = ["driver", "team", "grid", "finish_pos", "dnf"]
    result_frame = results[result_cols].copy()
    result_frame["finish_pos"] = pd.to_numeric(result_frame["finish_pos"], errors="coerce").astype("Int64")
    result_frame["grid"] = pd.to_numeric(result_frame["grid"], errors="coerce").astype("Int64")
    result_frame["dnf"] = _clean_boolean(result_frame["dnf"], "results.dnf")
    if result_frame["driver"].duplicated().any():
        raise FeatureBuildError("results contains duplicate driver rows")
    base = base.merge(result_frame, on="driver", how="left", validate="many_to_one")
    if base["team"].isna().any():
        missing_teams = base.loc[base["team"].isna(), "driver"].unique().tolist()
        raise FeatureBuildError(f"No results/team row found for drivers: {missing_teams}")

    # The leader's pace is looked up as-of the target driver's lap completion
    # time, using the leader identity reconstructed from positions at that time.
    pace_events = base[
        ["driver_number", "lap_end_utc", "rolling_pace_s"]
    ].dropna(subset=["rolling_pace_s"]).rename(
        columns={
            "driver_number": "leader_driver_number",
            "lap_end_utc": "leader_pace_event_utc",
            "rolling_pace_s": "leader_rolling_pace_s",
        }
    )
    pace_query = base[["row_id", "lap_end_utc", "leader_driver_number"]].dropna(
        subset=["leader_driver_number"]
    ).rename(columns={"lap_end_utc": "snapshot_utc"}).copy()
    if not pace_query.empty and not pace_events.empty:
        pace_query["leader_driver_number"] = pace_query["leader_driver_number"].astype("int64")
        pace_events["leader_driver_number"] = pace_events["leader_driver_number"].astype("int64")
        pace_match = pd.merge_asof(
            pace_query.sort_values("snapshot_utc", kind="mergesort"),
            pace_events.sort_values("leader_pace_event_utc", kind="mergesort"),
            left_on="snapshot_utc",
            right_on="leader_pace_event_utc",
            by="leader_driver_number",
            direction="backward",
            allow_exact_matches=True,
        )
        if (
            pace_match["leader_pace_event_utc"].notna()
            & (pace_match["leader_pace_event_utc"] > pace_match["snapshot_utc"])
        ).any():
            raise FeatureBuildError("Leader pace join unexpectedly used a future lap")
        base = base.merge(
            pace_match[["row_id", "leader_rolling_pace_s"]],
            on="row_id",
            how="left",
            validate="one_to_one",
        )
    else:
        base["leader_rolling_pace_s"] = np.nan
    base["pace_delta_to_leader_s"] = (
        pd.to_numeric(base["rolling_pace_s"], errors="coerce")
        - pd.to_numeric(base["leader_rolling_pace_s"], errors="coerce")
    )

    # Metadata and target labels.
    base["race_id"] = int(race_id)
    base["laps_remaining"] = total_laps - base["lap"]
    base["grid_position"] = base["grid"].astype("Int64")
    base["finish_position"] = base["finish_pos"].astype("Int64")
    finish = base["finish_position"]
    base["is_win"] = finish.eq(1).astype("boolean")
    base["is_top3"] = finish.le(3).where(finish.notna(), pd.NA).astype("boolean")
    base["is_top10"] = finish.le(10).where(finish.notna(), pd.NA).astype("boolean")
    base["is_dnf"] = base["dnf"].astype("boolean")
    base["circuit"] = str(race["circuit"])
    if season < 2022:
        era = "pre_2022"
    elif season <= 2025:
        era = "2022_2025_ground_effect"
    else:
        era = "2026_plus"
    base["era"] = era

    feature_columns = [
        "race_id",
        "driver",
        "lap",
        "laps_remaining",
        "position",
        "gap_to_leader_s",
        "laps_behind_leader",
        "gap_to_ahead_s",
        "gap_to_behind_s",
        "compound",
        "tyre_age",
        "stint",
        "stops_made",
        "track_status",
        "grid_position",
        "team",
        "rolling_pace_s",
        "pace_delta_to_leader_s",
        "air_temp",
        "track_temp",
        "rain",
        "circuit",
        "era",
        "finish_position",
        "is_win",
        "is_top3",
        "is_top10",
        "is_dnf",
    ]
    missing_feature_columns = sorted(set(feature_columns).difference(base.columns))
    if missing_feature_columns:
        raise FeatureBuildError(f"Internal builder did not create columns: {missing_feature_columns}")

    output = base[feature_columns].copy()
    output = output.sort_values(["race_id", "lap", "driver"], kind="mergesort").reset_index(drop=True)

    duplicate_keys = int(output.duplicated(["race_id", "driver", "lap"]).sum())
    if duplicate_keys:
        raise FeatureBuildError(f"Output has {duplicate_keys} duplicate (race_id, driver, lap) keys")
    if len(output) != len(laps):
        raise FeatureBuildError(
            f"Output row count {len(output)} differs from source completed laps {len(laps)}"
        )
    if output["position"].isna().any():
        raise FeatureBuildError("Output contains missing reconstructed positions")
    if not output["position"].between(1, len(base["driver_number"].unique())).all():
        raise FeatureBuildError("Output contains position outside the expected driver range")
    if output.loc[output["position"] == 1, "gap_to_ahead_s"].notna().any():
        raise FeatureBuildError("Leader rows must not have a gap_to_ahead_s value")
    if output["laps_remaining"].lt(0).any():
        raise FeatureBuildError("Output contains negative laps_remaining")

    diagnostics = {
        **position_diagnostics,
        "output_rows": len(output),
        "drivers": int(output["driver"].nunique()),
        "duplicate_primary_keys": duplicate_keys,
        "rows_with_grid_position": int(output["grid_position"].notna().sum()),
        "rows_with_rolling_pace": int(output["rolling_pace_s"].notna().sum()),
        "rows_with_pace_delta": int(output["pace_delta_to_leader_s"].notna().sum()),
        "rows_with_weather": int(output[["air_temp", "track_temp", "rain"]].notna().all(axis=1).sum()),
        "rows_with_missing_gap_to_leader": int(output["gap_to_leader_s"].isna().sum()),
        "rows_with_lap_deficit": int(output["laps_behind_leader"].gt(0).sum()),
        "rows_with_gap_to_ahead": int(output["gap_to_ahead_s"].notna().sum()),
        "rows_with_gap_to_behind": int(output["gap_to_behind_s"].notna().sum()),
        "stale_weather_rows": int((~weather_fresh).sum()),
        "future_weather_matches": int(
            (weather_match["date"] > weather_match["lap_end_utc"]).sum()
        ),
        "grid_positions_missing_in_source": int(result_frame["grid"].isna().sum()),
        "interval_max_age_limit_s": float(interval_max_age_s),
        "weather_max_age_limit_s": float(weather_max_age_s),
        "rolling_clean_lap_window": int(rolling_clean_laps),
        "era": era,
    }
    output.attrs["diagnostics"] = diagnostics
    return output
