import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data/raw/openf1/2024/bahrain_9472"

PROCESSED_DIR = ROOT / "data/processed/bahrain_2024"

OUTPUT_FILE = PROCESSED_DIR / "weather.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:

    logger.info("Starting weather transformation.")

    laps = pd.read_parquet(
        PROCESSED_DIR / "laps.parquet"
    )

    weather = pd.read_parquet(
        RAW_DIR / "weather.parquet"
    )

    # Parse timestamps consistently.
    laps["lap_end_utc"] = pd.to_datetime(
        laps["lap_end_utc"],
        format="ISO8601",
        utc=True,
    )

    weather["date_utc"] = pd.to_datetime(
        weather["date"],
        format="ISO8601",
        utc=True,
    )

    # Use the median completion time of all drivers
    # to represent each race lap.
    lap_reference = (
        laps.groupby("lap", as_index=False)
        .agg(lap_reference_utc=("lap_end_utc", "median"))
        .sort_values("lap_reference_utc")
    )

    weather = weather.sort_values("date_utc")

    # Select the latest weather reading at or before
    # the representative lap completion time.
    transformed = pd.merge_asof(
        lap_reference,
        weather[
            [
                "date_utc",
                "air_temperature",
                "track_temperature",
                "rainfall",
            ]
        ],
        left_on="lap_reference_utc",
        right_on="date_utc",
        direction="backward",
    )

    transformed["weather_age_s"] = (
        transformed["lap_reference_utc"]
        - transformed["date_utc"]
    ).dt.total_seconds()

    transformed = transformed.rename(
        columns={
            "air_temperature": "air_temp",
            "track_temperature": "track_temp",
            "rainfall": "rain",
        }
    )

    transformed["rain"] = (
        transformed["rain"]
        .map({0: False, 1: True})
        .astype("boolean")
    )

    result = transformed[
        [
            "lap",
            "air_temp",
            "track_temp",
            "rain",
            "weather_age_s",
        ]
    ]

    # Validation.
    if result["lap"].duplicated().any():
        raise ValueError("Duplicate lap weather records found.")

    if len(result) != 57:
        raise ValueError(
            f"Expected 57 weather records, got {len(result)}"
        )

    if result["air_temp"].isna().any():
        raise ValueError("Missing air temperature detected.")

    if result["track_temp"].isna().any():
        raise ValueError("Missing track temperature detected.")

    if result["weather_age_s"].isna().any():
        raise ValueError("Some laps have no previous weather observation.")

    if (result["weather_age_s"] < 0).any():
        raise ValueError("Future weather observation detected.")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
        compression="snappy",
    )

    logger.info("Weather transformation completed.")
    logger.info("Records: %s", len(result))
    logger.info("Average weather age: %.2f seconds",
                result["weather_age_s"].mean())
    logger.info("Maximum weather age: %.2f seconds",
                result["weather_age_s"].max())
    logger.info("Saved: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
