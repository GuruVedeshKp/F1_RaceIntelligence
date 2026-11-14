from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data/raw/openf1/2024/bahrain_9472"

PROCESSED_DIR = ROOT / "data/processed/bahrain_2024"

OUTPUT_FILE = PROCESSED_DIR / "results.parquet"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_data(name: str) -> pd.DataFrame:
    """Load a raw OpenF1 dataset."""

    path = RAW_DIR / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_parquet(path)


def main() -> None:

    logger.info("Starting Bahrain results transformation.")

    results = load_data("session_result")
    drivers = load_data("drivers")

    # --------------------------------------------------
    # 1. DRIVER MAPPING
    # --------------------------------------------------

    driver_map = drivers[
        [
            "driver_number",
            "name_acronym",
            "team_name",
        ]
    ].drop_duplicates()

    if driver_map["driver_number"].duplicated().any():
        raise ValueError("Duplicate driver mappings detected.")

    results = results.merge(
        driver_map,
        on="driver_number",
        how="left",
        validate="one_to_one",
    )

    if results["name_acronym"].isna().any():
        raise ValueError("Some results have no driver mapping.")

    # --------------------------------------------------
    # 2. NORMALISE DATA TYPES
    # --------------------------------------------------

    results["finish_pos"] = pd.to_numeric(
        results["position"],
        errors="coerce",
    ).astype("Int64")

    results["number_of_laps"] = pd.to_numeric(
        results["number_of_laps"],
        errors="coerce",
    ).astype("Int64")

    results["points"] = pd.to_numeric(
        results["points"],
        errors="coerce",
    ).astype("Float64")

    results["duration_s"] = pd.to_numeric(
        results["duration"],
        errors="coerce",
    ).astype("Float64")

    results["driver_number"] = (
        results["driver_number"].astype("int64")
    )

    # Preserve mixed classification gaps as source text.
    results["gap_to_leader_raw"] = (
        results["gap_to_leader"].astype("string")
    )

    # --------------------------------------------------
    # 3. DETERMINE FINISHING STATUS
    # --------------------------------------------------

    def determine_status(row: pd.Series) -> str:

        if bool(row["dsq"]):
            return "DSQ"

        if bool(row["dns"]):
            return "DNS"

        if bool(row["dnf"]):
            return "DNF"

        return "FINISHED"

    results["status"] = results.apply(
        determine_status,
        axis=1,
    )

    # --------------------------------------------------
    # 4. STARTING GRID
    # --------------------------------------------------

    # Actual starting-grid positions have not been
    # independently retrieved and verified yet.
    # Do not substitute finishing positions.

    results["grid"] = pd.Series(
        pd.NA,
        index=results.index,
        dtype="Int64",
    )

    # --------------------------------------------------
    # 5. BUILD CANONICAL RESULT DATASET
    # --------------------------------------------------

    clean_results = results[
        [
            "meeting_key",
            "session_key",
            "driver_number",
            "name_acronym",
            "team_name",
            "grid",
            "finish_pos",
            "number_of_laps",
            "status",
            "points",
            "duration_s",
            "gap_to_leader_raw",
            "dnf",
            "dns",
            "dsq",
        ]
    ].rename(
        columns={
            "name_acronym": "driver",
            "team_name": "team",
        }
    )

    clean_results = clean_results.sort_values(
        "finish_pos",
        na_position="last",
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 6. VALIDATION
    # --------------------------------------------------

    if len(clean_results) != 20:
        raise ValueError(
            f"Expected 20 drivers, received {len(clean_results)}."
        )

    if clean_results["driver"].duplicated().any():
        raise ValueError("Duplicate driver results detected.")

    if clean_results["finish_pos"].isna().any():
        raise ValueError("Missing finishing positions detected.")

    if clean_results["finish_pos"].duplicated().any():
        raise ValueError("Duplicate finishing positions detected.")

    if clean_results["points"].isna().any():
        raise ValueError("Missing championship points detected.")

    if not clean_results["status"].isin(
        ["FINISHED", "DNF", "DNS", "DSQ"]
    ).all():
        raise ValueError("Unexpected finishing status found.")

    # --------------------------------------------------
    # 7. SAVE
    # --------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_results.to_parquet(
        OUTPUT_FILE,
        index=False,
        compression="snappy",
    )

    logger.info("Results transformation completed.")
    logger.info("Total drivers: %s", len(clean_results))
    logger.info(
        "Finished: %s",
        clean_results["status"].eq("FINISHED").sum(),
    )
    logger.info(
        "DNF: %s",
        clean_results["status"].eq("DNF").sum(),
    )
    logger.info(
        "DNS: %s",
        clean_results["status"].eq("DNS").sum(),
    )
    logger.info(
        "DSQ: %s",
        clean_results["status"].eq("DSQ").sum(),
    )
    logger.info(
        "Total points: %.1f",
        clean_results["points"].sum(),
    )
    logger.info(
        "Missing race durations: %s",
        clean_results["duration_s"].isna().sum(),
    )
    logger.info(
        "Starting-grid positions pending: %s",
        clean_results["grid"].isna().sum(),
    )

    logger.info("Saved: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
