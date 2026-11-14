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

OUTPUT_DIR = ROOT / "data" / "processed" / "bahrain_2024"

OUTPUT_FILE = OUTPUT_DIR / "laps.parquet"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data(name: str) -> pd.DataFrame:
    """Load a raw OpenF1 Parquet dataset."""

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
# CLASSIFY RACE CONTROL EVENTS
# --------------------------------------------------

STATUS_PRIORITY = {
    "YELLOW": 1,
    "DOUBLE_YELLOW": 2,
    "VSC": 3,
    "SC": 4,
    "RED": 5,
}



def classify_event(row: pd.Series) -> str | None:
    """Classify relevant race-control messages safely."""

    message = str(row.get("message") or "").upper().strip()
    flag = str(row.get("flag") or "").upper().strip()
    category = str(row.get("category") or "").upper().strip()

    # Chequered flag indicates the finish, not a red flag.
    if flag == "CHEQUERED" or message.startswith("CHEQUERED"):
        return None

    # Safety car and virtual safety car.
    if "VIRTUAL SAFETY CAR" in message or category == "SAFETYCAR":
        return "VSC"

    if "SAFETY CAR" in message:
        return "SC"

    # Match actual red-flag announcements.
    if flag in {"RED", "RED FLAG"} or message.startswith("RED FLAG"):
        return "RED"

    # Yellow flags.
    if flag == "DOUBLE YELLOW" or "DOUBLE YELLOW" in message:
        return "DOUBLE_YELLOW"

    if flag == "YELLOW" or message.startswith("YELLOW"):
        return "YELLOW"

    return None



def build_flag_map(race_control: pd.DataFrame) -> dict[int, str]:
    """Build a conservative lap-to-track-status mapping."""

    events = race_control.copy()

    events["track_status"] = events.apply(
        classify_event,
        axis=1,
    )

    events = events.dropna(
        subset=["lap_number", "track_status"]
    )

    status_by_lap = {}

    for lap, group in events.groupby("lap_number"):

        statuses = group["track_status"].tolist()

        selected = max(
            statuses,
            key=lambda status: STATUS_PRIORITY[status],
        )

        status_by_lap[int(lap)] = selected

    return status_by_lap


# --------------------------------------------------
# TRANSFORM LAP DATA
# --------------------------------------------------

def main() -> None:

    logger.info("Starting Bahrain lap transformation.")

    laps = load_data("laps")
    drivers = load_data("drivers")
    stints = load_data("stints")
    positions = load_data("position")
    pit = load_data("pit")
    race_control = load_data("race_control")

    logger.info("Loaded all six source datasets.")

    # ----------------------------------------------
    # 1. DRIVER MAPPING
    # ----------------------------------------------

    driver_map = drivers[
        ["driver_number", "name_acronym"]
    ].drop_duplicates()

    if driver_map["driver_number"].duplicated().any():
        raise ValueError("Duplicate driver-number mapping found.")

    laps = laps.merge(
        driver_map,
        on="driver_number",
        how="left",
        validate="many_to_one",
    )

    if laps["name_acronym"].isna().any():
        raise ValueError("Some laps could not be mapped to drivers.")

    # ----------------------------------------------
    # 2. LAP TIMESTAMPS
    # ----------------------------------------------

    laps["date_start_utc"] = parse_utc(
        laps["date_start"]
    )

    laps["lap_duration"] = pd.to_numeric(
        laps["lap_duration"],
        errors="coerce",
    )

    laps["lap_end_utc"] = (
        laps["date_start_utc"]
        + pd.to_timedelta(
            laps["lap_duration"],
            unit="s",
        )
    )

    # ----------------------------------------------
    # 3. TYRE STINT MAPPING
    # ----------------------------------------------

    stint_columns = [
        "driver_number",
        "stint_number",
        "lap_start",
        "lap_end",
        "compound",
        "tyre_age_at_start",
    ]

    candidates = laps[
        ["driver_number", "lap_number"]
    ].merge(
        stints[stint_columns],
        on="driver_number",
        how="left",
    )

    matches = candidates.loc[
        candidates["lap_start"].notna()
        & candidates["lap_end"].notna()
        & (
            candidates["lap_number"]
            >= candidates["lap_start"]
        )
        & (
            candidates["lap_number"]
            <= candidates["lap_end"]
        )
    ].copy()

    duplicate_stint_matches = matches.duplicated(
        ["driver_number", "lap_number"],
        keep=False,
    )

    if duplicate_stint_matches.any():
        raise ValueError(
            "Overlapping tyre stints found for one or more laps."
        )

    stint_lookup = matches[
        [
            "driver_number",
            "lap_number",
            "stint_number",
            "compound",
            "tyre_age_at_start",
            "lap_start",
        ]
    ]

    laps = laps.merge(
        stint_lookup,
        on=["driver_number", "lap_number"],
        how="left",
        validate="one_to_one",
    )

    # Tyre age is measured at the start of the recorded lap.
    # This aligns the tyre age with the lap-time measurement.
    laps["tyre_age"] = (
        pd.to_numeric(
            laps["tyre_age_at_start"],
            errors="coerce",
        )
        + laps["lap_number"]
        - pd.to_numeric(
            laps["lap_start"],
            errors="coerce",
        )
    )

    # ----------------------------------------------
    # 4. PIT STOP MAPPING
    # ----------------------------------------------

    pit_keys = set(
        zip(
            pit["driver_number"].astype(int),
            pit["lap_number"].astype(int),
        )
    )

    laps["pit_in"] = [
        (int(driver), int(lap)) in pit_keys
        for driver, lap in zip(
            laps["driver_number"],
            laps["lap_number"],
        )
    ]

    laps["pit_out"] = (
        laps["is_pit_out_lap"]
        .fillna(False)
        .astype(bool)
    )

    # ----------------------------------------------
    # 5. POSITION MAPPING
    # ----------------------------------------------

    positions["date_utc"] = parse_utc(
        positions["date"]
    )

    positions = positions[
        [
            "driver_number",
            "date_utc",
            "position",
        ]
    ].dropna(
        subset=["date_utc", "position"]
    )

    positions = positions.sort_values(
        ["date_utc", "driver_number"]
    )

    laps = laps.sort_values(
        ["lap_end_utc", "driver_number"]
    )

    # Use the latest position event at or before
    # the completion of each recorded lap.
    laps = pd.merge_asof(
        laps,
        positions,
        left_on="lap_end_utc",
        right_on="date_utc",
        by="driver_number",
        direction="backward",
        allow_exact_matches=True,
    )

    laps["position_state_age_s"] = (
        laps["lap_end_utc"] - laps["date_utc"]
    ).dt.total_seconds()

    # ----------------------------------------------
    # 6. RACE CONTROL / TRACK STATUS
    # ----------------------------------------------

    flag_map = build_flag_map(race_control)

    laps["track_status"] = (
        laps["lap_number"]
        .map(flag_map)
    )

    # ----------------------------------------------
    # 7. CLEAN LAP CLASSIFICATION
    # ----------------------------------------------

    wet_compounds = {"INTERMEDIATE", "WET"}

    wet_laps = (
        laps["compound"]
        .astype("string")
        .str.upper()
        .isin(wet_compounds)
    )

    flagged_laps = laps["track_status"].notna()

    laps["is_clean"] = (
        (laps["lap_number"] > 1)
        & (~laps["pit_in"])
        & (~laps["pit_out"])
        & (~flagged_laps)
        & (~wet_laps)
    )

    # ----------------------------------------------
    # 8. BUILD CANONICAL OUTPUT
    # ----------------------------------------------

    clean_laps = pd.DataFrame({
        "meeting_key": laps["meeting_key"],
        "session_key": laps["session_key"],
        "driver_number": laps["driver_number"],
        "driver": laps["name_acronym"],
        "lap": laps["lap_number"],
        "lap_start_utc": laps["date_start_utc"],
        "lap_end_utc": laps["lap_end_utc"],
        "lap_time_s": laps["lap_duration"],
        "position": laps["position"],
        "compound": laps["compound"],
        "tyre_age": laps["tyre_age"],
        "stint": laps["stint_number"],
        "pit_in": laps["pit_in"],
        "pit_out": laps["pit_out"],
        "track_status": laps["track_status"],
        "is_clean": laps["is_clean"],
        "position_state_age_s": laps["position_state_age_s"],
    })

    clean_laps["position"] = clean_laps[
        "position"
    ].astype("Int64")

    clean_laps["stint"] = clean_laps[
        "stint"
    ].astype("Int64")

    clean_laps["tyre_age"] = pd.to_numeric(
        clean_laps["tyre_age"],
        errors="coerce",
    ).astype("Float64")

    # ----------------------------------------------
    # 9. VALIDATION
    # ----------------------------------------------

    duplicate_count = clean_laps.duplicated(
        ["driver", "lap"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate driver-lap rows."
        )

    if clean_laps["lap_time_s"].isna().any():
        logger.warning("Some lap times are missing.")

    logger.info("Transformation completed.")
    logger.info("Total lap records: %s", len(clean_laps))
    logger.info("Unique drivers: %s", clean_laps["driver"].nunique())

    logger.info(
        "Missing position: %s",
        clean_laps["position"].isna().sum(),
    )

    logger.info(
        "Missing tyre compound: %s",
        clean_laps["compound"].isna().sum(),
    )

    logger.info(
        "Missing tyre age: %s",
        clean_laps["tyre_age"].isna().sum(),
    )

    logger.info(
        "Pit-in laps: %s",
        clean_laps["pit_in"].sum(),
    )

    logger.info(
        "Pit-out laps: %s",
        clean_laps["pit_out"].sum(),
    )

    logger.info(
        "Clean laps: %s",
        clean_laps["is_clean"].sum(),
    )

    logger.info(
        "Track-status distribution:\n%s",
        clean_laps["track_status"].value_counts(dropna=False),
    )

    # ----------------------------------------------
    # 10. SAVE OUTPUT
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_laps.to_parquet(
        OUTPUT_FILE,
        index=False,
        compression="snappy",
    )

    logger.info("Clean lap dataset saved to: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
