from __future__ import annotations

from datetime import timedelta

import pandas as pd
from pandas.testing import assert_frame_equal

from f1ri.features.lap_state import build_lap_state

INPUT_FEATURE_COLUMNS = [
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
]


def _make_inputs():
    """Create a small four-lap race with events before and after a cutoff."""
    start = pd.Timestamp("2024-03-02T10:00:00Z")
    driver_specs = [
        (1, "VER", 0, 1),
        (2, "LEC", 2, 2),
    ]

    lap_records = []
    for driver_number, driver, offset_s, initial_position in driver_specs:
        for lap in range(1, 5):
            lap_end = start + timedelta(
                minutes=5 * int(lap),
                seconds=int(offset_s),
            )
            position = (
                initial_position
                if lap <= 3
                else (2 if initial_position == 1 else 1)
            )
            lap_records.append(
                {
                    "driver_number": driver_number,
                    "driver": driver,
                    "lap": lap,
                    "lap_end_utc": lap_end.isoformat(),
                    "lap_time_s": float(100 - lap + driver_number - 1),
                    "position": position,
                    "compound": "SOFT",
                    "tyre_age": lap - 1,
                    "stint": 1,
                    "pit_in": False,
                    "pit_out": False,
                    "track_status": None,
                    "is_clean": True,
                }
            )

    laps = pd.DataFrame(lap_records)

    positions = pd.DataFrame(
        [
            {
                "driver_number": 1,
                "date": "2024-03-02T10:00:00Z",
                "position": 1,
            },
            {
                "driver_number": 2,
                "date": "2024-03-02T10:00:00Z",
                "position": 2,
            },
            {
                "driver_number": 1,
                "date": "2024-03-02T10:18:00Z",
                "position": 2,
            },
            {
                "driver_number": 2,
                "date": "2024-03-02T10:18:00Z",
                "position": 1,
            },
        ]
    )

    intervals = pd.DataFrame(
        [
            {
                "driver_number": 1,
                "date": "2024-03-02T10:04:59Z",
                "gap_to_leader": 0.0,
                "interval": 0.0,
            },
            {
                "driver_number": 2,
                "date": "2024-03-02T10:04:59Z",
                "gap_to_leader": 1.5,
                "interval": 1.5,
            },
            {
                "driver_number": 1,
                "date": "2024-03-02T10:18:00Z",
                "gap_to_leader": 0.0,
                "interval": 0.0,
            },
            {
                "driver_number": 2,
                "date": "2024-03-02T10:18:00Z",
                "gap_to_leader": 1.2,
                "interval": 1.2,
            },
        ]
    )

    weather = pd.DataFrame(
        [
            {
                "date": "2024-03-02T10:04:30Z",
                "air_temperature": 25.0,
                "track_temperature": 35.0,
                "rainfall": 0,
            },
            {
                "date": "2024-03-02T10:09:30Z",
                "air_temperature": 25.5,
                "track_temperature": 35.5,
                "rainfall": 0,
            },
            {
                "date": "2024-03-02T10:14:30Z",
                "air_temperature": 26.0,
                "track_temperature": 36.0,
                "rainfall": 0,
            },
            {
                "date": "2024-03-02T10:18:00Z",
                "air_temperature": 99.0,
                "track_temperature": 99.0,
                "rainfall": 1,
            },
        ]
    )

    pit_stops = pd.DataFrame(
        [
            {
                "driver_number": 1,
                "pit_time_utc": "2024-03-02T10:18:00Z",
            }
        ]
    )

    results = pd.DataFrame(
        [
            {
                "driver": "VER",
                "team": "Red Bull",
                "grid": 1,
                "finish_pos": 1,
                "dnf": False,
            },
            {
                "driver": "LEC",
                "team": "Ferrari",
                "grid": 2,
                "finish_pos": 2,
                "dnf": False,
            },
        ]
    )

    race = {
        "season": 2024,
        "circuit": "Test Circuit",
        "total_laps": 4,
    }

    return race, laps, positions, intervals, weather, pit_stops, results


def _build(inputs) -> pd.DataFrame:
    race, laps, positions, intervals, weather, pit_stops, results = inputs
    return build_lap_state(
        race_id=999,
        race=race,
        laps=laps,
        positions=positions,
        intervals=intervals,
        weather=weather,
        pit_stops=pit_stops,
        results=results,
    )


def test_future_observations_do_not_change_earlier_features() -> None:
    original = _make_inputs()
    changed = [
        item.copy(deep=True) if isinstance(item, pd.DataFrame) else dict(item)
        for item in original
    ]
    _, laps, positions, intervals, weather, pit_stops, _ = changed

    # Change only completed-lap data and source events after the lap-3 cutoff.
    future_laps = laps["lap"].eq(4)
    laps.loc[future_laps, "lap_time_s"] = 250.0
    laps.loc[future_laps & laps["driver_number"].eq(1), "position"] = 1
    laps.loc[future_laps & laps["driver_number"].eq(2), "position"] = 2

    future_position = positions["date"].eq("2024-03-02T10:18:00Z")
    positions.loc[future_position, "position"] = [1, 2]

    future_interval = intervals["date"].eq("2024-03-02T10:18:00Z")
    intervals.loc[future_interval, "gap_to_leader"] = [0.0, 20.0]
    intervals.loc[future_interval, "interval"] = [0.0, 20.0]

    future_weather = weather["date"].eq("2024-03-02T10:18:00Z")
    weather.loc[future_weather, "air_temperature"] = 99.0
    weather.loc[future_weather, "track_temperature"] = 99.0
    weather.loc[future_weather, "rainfall"] = 1

    pit_stops.loc[:, "pit_time_utc"] = "2024-03-02T10:19:00Z"

    before = _build(original)
    after = _build(tuple(changed))

    before_early = (
        before.loc[before["lap"].le(3), INPUT_FEATURE_COLUMNS]
        .sort_values(["driver", "lap"])
        .reset_index(drop=True)
    )
    after_early = (
        after.loc[after["lap"].le(3), INPUT_FEATURE_COLUMNS]
        .sort_values(["driver", "lap"])
        .reset_index(drop=True)
    )

    assert_frame_equal(
        before_early,
        after_early,
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-10,
    )


def test_rolling_pace_at_lap_three_does_not_use_lap_four() -> None:
    original = _make_inputs()
    changed = [
        item.copy(deep=True) if isinstance(item, pd.DataFrame) else dict(item)
        for item in original
    ]
    changed[1].loc[changed[1]["lap"].eq(4), "lap_time_s"] = 999.0

    before = _build(original)
    after = _build(tuple(changed))

    before_lap3 = (
        before.loc[before["lap"].eq(3), ["driver", "rolling_pace_s"]]
        .sort_values("driver")
        .reset_index(drop=True)
    )
    after_lap3 = (
        after.loc[after["lap"].eq(3), ["driver", "rolling_pace_s"]]
        .sort_values("driver")
        .reset_index(drop=True)
    )

    assert_frame_equal(
        before_lap3,
        after_lap3,
        check_dtype=False,
        check_exact=True,
    )


def test_position_and_weather_at_lap_three_use_only_prior_events() -> None:
    result = _build(_make_inputs())
    lap3 = result.loc[result["lap"].eq(3)]

    assert lap3["position"].eq(1).sum() == 1
    assert lap3["position"].eq(2).sum() == 1
    assert lap3["air_temp"].eq(26.0).all()
    assert lap3["track_temp"].eq(36.0).all()
    assert lap3["rain"].eq(False).all()

    diagnostics = result.attrs["diagnostics"]
    assert diagnostics["future_position_matches"] == 0
    assert diagnostics["future_interval_matches"] == 0
    assert diagnostics["future_weather_matches"] == 0
