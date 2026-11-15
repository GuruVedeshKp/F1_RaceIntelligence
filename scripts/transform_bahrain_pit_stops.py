from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    ROOT / "data" / "raw" / "openf1" / "2024" / "bahrain_9472"
)

PROCESSED_DIR = ROOT / "data" / "processed" / "bahrain_2024"

OUTPUT_FILE = PROCESSED_DIR / "pit_stops.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def load_data(name: str) -> pd.DataFrame:
    """Load a raw OpenF1 dataset."""

    path = RAW_DIR / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_parquet(path)


def parse_utc(series: pd.Series) -> pd.Series:
    """Parse ISO8601 timestamps with mixed precision."""

    return pd.to_datetime(
        series,
        format="ISO8601",
        utc=True,
        errors="coerce",
    )


# --------------------------------------------------
# TRANSFORMATION
# --------------------------------------------------

def main() -> None:

    logger.info("Starting Bahrain pit-stop transformation.")

    pit = load_data("pit")
    drivers = load_data("drivers")
    race_control = load_data("race_control")

    logger.info("Loaded pit, driver and race-control datasets.")

    # ----------------------------------------------
    # 1. VALIDATE SOURCE DATA
    # ----------------------------------------------

    required_columns = {
        "driver_number",
        "lap_number",
        "date",
        "lane_duration",
        "stop_duration",
    }

    missing_columns = required_columns - set(pit.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required pit columns: {missing_columns}"
        )

    # ----------------------------------------------
    # 2. MAP DRIVER INFORMATION
    # ----------------------------------------------

    driver_map = drivers[
        ["driver_number", "name_acronym"]
    ].drop_duplicates()

    if driver_map["driver_number"].duplicated().any():
        raise ValueError("Duplicate driver mappings found.")

    pit = pit.merge(
        driver_map,
        on="driver_number",
        how="left",
        validate="many_to_one",
    )

    if pit["name_acronym"].isna().any():
        raise ValueError("Some pit stops have unknown drivers.")

    # ----------------------------------------------
    # 3. NORMALISE DATA TYPES
    # ----------------------------------------------

    pit["driver_number"] = pd.to_numeric(
        pit["driver_number"],
        errors="raise",
    ).astype("int64")

    pit["lap_number"] = pd.to_numeric(
        pit["lap_number"],
        errors="raise",
    ).astype("int64")

    pit["date_utc"] = parse_utc(pit["date"])

    pit["lane_duration_s"] = pd.to_numeric(
        pit["lane_duration"],
        errors="coerce",
    ).astype("Float64")

    pit["stationary_duration_s"] = pd.to_numeric(
        pit["stop_duration"],
        errors="coerce",
    ).astype("Float64")

    if pit["date_utc"].isna().any():
        raise ValueError("Invalid pit-stop timestamps found.")

    if pit["lane_duration_s"].isna().any():
        raise ValueError("Missing pit-lane duration found.")

    # ----------------------------------------------
    # 4. VALIDATE DEPRECATED PIT DURATION FIELD
    # ----------------------------------------------

    if "pit_duration" in pit.columns:

        legacy_duration = pd.to_numeric(
            pit["pit_duration"],
            errors="coerce",
        )

        comparable = pit["lane_duration_s"].notna() & legacy_duration.notna()

        differences = (
            pit.loc[comparable, "lane_duration_s"].astype(float)
            - legacy_duration.loc[comparable].astype(float)
        ).abs()

        if not differences.le(0.001).all():
            raise ValueError(
                "pit_duration and lane_duration do not match."
            )

        logger.info(
            "Legacy pit_duration agrees with lane_duration."
        )

    # ----------------------------------------------
    # 5. CHECK SAFETY CAR CONDITIONS
    # ----------------------------------------------

    messages = race_control["message"].astype("string")

    category = race_control["category"].astype("string")

    sc_mask = (
        messages.str.contains(
            r"SAFETY CAR|VIRTUAL SAFETY CAR|\bVSC\b",
            case=False,
            na=False,
        )
        |
        category.str.contains(
            "SafetyCar",
            case=False,
            na=False,
        )
    )

    if sc_mask.any():
        raise ValueError(
            "Safety-car-related events detected. "
            "Under-SC pit-stop classification needs "
            "event-time analysis before proceeding."
        )

    # No SC/VSC messages were found in this Bahrain
    # race-control dataset, so these stops are not
    # classified as under a safety car.
    pit["under_sc"] = False

    # ----------------------------------------------
    # 6. ESTIMATED PIT LOSS
    # ----------------------------------------------

    # Pit loss is not equivalent to pit-lane duration.
    # It will be estimated during Week 6.
    pit["pit_loss_s"] = pd.Series(
        pd.NA,
        index=pit.index,
        dtype="Float64",
    )

    # ----------------------------------------------
    # 7. BUILD CANONICAL DATASET
    # ----------------------------------------------

    result = pit[
        [
            "meeting_key",
            "session_key",
            "driver_number",
            "name_acronym",
            "lap_number",
            "date_utc",
            "lane_duration_s",
            "stationary_duration_s",
            "pit_loss_s",
            "under_sc",
        ]
    ].rename(
        columns={
            "name_acronym": "driver",
            "lap_number": "lap",
            "date_utc": "pit_time_utc",
        }
    )

    result = result.sort_values(
        ["driver", "lap"]
    ).reset_index(drop=True)

    # ----------------------------------------------
    # 8. DATA QUALITY CHECKS
    # ----------------------------------------------

    duplicate_count = result.duplicated(
        subset=["driver_number", "lap"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate driver/lap stops."
        )

    if result["lane_duration_s"].le(0).any():
        raise ValueError("Non-positive pit-lane duration found.")

    logger.info("Transformation completed.")
    logger.info("Total pit stops: %s", len(result))
    logger.info("Unique drivers: %s", result["driver"].nunique())
    logger.info(
        "Missing stationary durations: %s",
        result["stationary_duration_s"].isna().sum(),
    )
    logger.info(
        "Missing pit-loss estimates: %s",
        result["pit_loss_s"].isna().sum(),
    )
    logger.info(
        "Stops under SC/VSC: %s",
        result["under_sc"].sum(),
    )

    # ----------------------------------------------
    # 9. SAVE PARQUET
    # ----------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
        compression="snappy",
    )

    logger.info("Saved dataset: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
