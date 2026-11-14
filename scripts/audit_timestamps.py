from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    ROOT / "data" / "raw" / "openf1" / "2024" / "bahrain_9472"
)


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


def parse_dates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(
        df[column],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )
    return df


def main() -> None:
    sessions = load("sessions")
    laps = parse_dates(load("laps"), "date_start")
    positions = parse_dates(load("position"), "date")
    intervals = parse_dates(load("intervals"), "date")
    weather = parse_dates(load("weather"), "date")
    pit = parse_dates(load("pit"), "date")
    race_control = parse_dates(load("race_control"), "date")

    session_start = pd.to_datetime(
        sessions.iloc[0]["date_start"],
        format="ISO8601",
        utc=True,
    )

    session_end = pd.to_datetime(
        sessions.iloc[0]["date_end"],
        format="ISO8601",
        utc=True,
    )

    print("\nSESSION WINDOW")
    print("Start:", session_start)
    print("End:", session_end)

    datasets = {
        "laps": (laps, "date_start"),
        "positions": (positions, "date"),
        "intervals": (intervals, "date"),
        "weather": (weather, "date"),
        "pit": (pit, "date"),
        "race_control": (race_control, "date"),
    }

    print("\nTIMESTAMP RANGES")

    for name, (df, column) in datasets.items():
        valid = df[column].dropna()

        if valid.empty:
            print(f"\n{name}: No valid timestamps")
            continue

        print(f"\n{name.upper()}")
        print("First:", valid.min())
        print("Last:", valid.max())
        print("Before session:", (valid < session_start).sum())
        print("After session:", (valid > session_end).sum())

        print(
            "Offset of first record from session start:",
            round((valid.min() - session_start).total_seconds(), 2),
            "seconds",
        )

    # Calculate lap completion timestamps.
    laps["lap_duration"] = pd.to_numeric(
        laps["lap_duration"],
        errors="coerce",
    )

    laps["lap_end"] = (
        laps["date_start"]
        + pd.to_timedelta(laps["lap_duration"], unit="s")
    )

    print("\nLAP COMPLETION RANGE")
    print("First lap completion:", laps["lap_end"].min())
    print("Last lap completion:", laps["lap_end"].max())

    # Check how old the latest available position
    # observation is at each driver's lap completion.
    print("\nPOSITION AS-OF ALIGNMENT")

    position_ages = []

    for driver in laps["driver_number"].unique():

        driver_laps = laps[
            laps["driver_number"] == driver
        ][["lap_number", "lap_end"]].dropna().sort_values("lap_end")

        driver_positions = positions[
            positions["driver_number"] == driver
        ][["date", "position"]].dropna().sort_values("date")

        if driver_positions.empty:
            print(f"Driver {driver}: No position observations")
            continue

        joined = pd.merge_asof(
            driver_laps,
            driver_positions,
            left_on="lap_end",
            right_on="date",
            direction="backward",
        )

        ages = (
            joined["lap_end"] - joined["date"]
        ).dt.total_seconds()

        position_ages.extend(ages.dropna().tolist())

    if position_ages:
        age_series = pd.Series(position_ages)

        print("Matched lap records:", len(age_series))
        print("Median observation age:", round(age_series.median(), 2), "seconds")
        print("95th percentile age:", round(age_series.quantile(0.95), 2), "seconds")
        print("Older than 180 seconds:", (age_series > 180).sum())

    # Check weather alignment against the representative
    # lap-completion timestamp for each race lap.
    print("\nWEATHER AS-OF ALIGNMENT")

    lap_reference = (
        laps.groupby("lap_number", as_index=False)["lap_end"]
        .median()
        .sort_values("lap_end")
    )

    weather_sorted = (
        weather[["date", "air_temperature", "track_temperature"]]
        .dropna(subset=["date"])
        .sort_values("date")
    )

    weather_join = pd.merge_asof(
        lap_reference,
        weather_sorted,
        left_on="lap_end",
        right_on="date",
        direction="backward",
    )

    weather_age = (
        weather_join["lap_end"] - weather_join["date"]
    ).dt.total_seconds()

    print("Race lap numbers:", len(weather_join))
    print("Matched weather readings:", weather_age.notna().sum())
    print("Median weather age:", round(weather_age.median(), 2), "seconds")
    print("Maximum weather age:", round(weather_age.max(), 2), "seconds")

    # Search every race-control message for safety-car references.
    print("\nSAFETY CAR MESSAGE CHECK")

    matches = race_control[
        race_control["message"].str.contains(
            r"SAFETY CAR|VSC|VIRTUAL SAFETY",
            case=False,
            na=False,
        )
    ]

    if matches.empty:
        print("No safety-car-related messages found.")
    else:
        print(
            matches[["date", "lap_number", "category", "message"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
