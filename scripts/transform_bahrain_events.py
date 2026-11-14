from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data/raw/openf1/2024/bahrain_9472"

PROCESSED_DIR = ROOT / "data/processed/bahrain_2024"

OUTPUT_FILE = PROCESSED_DIR / "track_events.parquet"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_data(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    return pd.read_parquet(path)


def safe_text(value) -> str:
    """Safely convert nullable source values into uppercase text."""

    if pd.isna(value):
        return ""

    return str(value).upper().strip()


def classify_event(row: pd.Series) -> str | None:
    """Classify relevant race-control messages."""

    message = safe_text(row.get("message"))
    flag = safe_text(row.get("flag"))
    category = safe_text(row.get("category"))

    # Finishing flag is not a red flag.
    if flag == "CHEQUERED" or message.startswith("CHEQUERED"):
        return None

    # Safety car classifications.
    if "VIRTUAL SAFETY CAR" in message or flag == "VSC":
        return "VSC"

    if "SAFETY CAR" in message or category == "SAFETYCAR":
        return "SC"

    # Red flag.
    if flag in {"RED", "RED FLAG"} or message.startswith("RED FLAG"):
        return "RED"

    # Yellow flags.
    if flag == "DOUBLE YELLOW" or message.startswith("DOUBLE YELLOW"):
        return "YELLOW"

    if flag == "YELLOW" or message.startswith("YELLOW"):
        return "YELLOW"

    # Status transitions.
    if flag == "GREEN" or message.startswith("GREEN"):
        return "GREEN"

    if flag == "CLEAR":
        return "CLEAR"

    return None


def main() -> None:

    logger.info("Starting track-event transformation.")

    race_control = load_data("race_control")

    logger.info(
        "Loaded %s race-control messages.",
        len(race_control),
    )

    # --------------------------------------------------
    # 1. CLASSIFY EVENTS
    # --------------------------------------------------

    events = race_control.copy()

    events["event_type"] = events.apply(
        classify_event,
        axis=1,
    )

    events = events.dropna(
        subset=["event_type"]
    ).copy()

    # --------------------------------------------------
    # 2. NORMALISE TIMESTAMPS
    # --------------------------------------------------

    events["event_time_utc"] = pd.to_datetime(
        events["date"],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )

    if events["event_time_utc"].isna().any():
        raise ValueError("Invalid event timestamps detected.")

    # --------------------------------------------------
    # 3. NORMALISE LAP NUMBERS
    # --------------------------------------------------

    events["start_lap"] = pd.to_numeric(
        events["lap_number"],
        errors="coerce",
    ).astype("Int64")

    if events["start_lap"].isna().any():
        raise ValueError("Missing event lap numbers detected.")

    # Event messages do not directly establish
    # an end lap. Keep it NULL rather than guessing.
    events["end_lap"] = pd.Series(
        pd.NA,
        index=events.index,
        dtype="Int64",
    )

    # --------------------------------------------------
    # 4. NORMALISE OPTIONAL FIELDS
    # --------------------------------------------------

    events["driver_number"] = pd.to_numeric(
        events["driver_number"],
        errors="coerce",
    ).astype("Int64")

    events["sector"] = pd.to_numeric(
        events["sector"],
        errors="coerce",
    ).astype("Int64")

    # --------------------------------------------------
    # 5. BUILD CANONICAL DATASET
    # --------------------------------------------------

    result = events[
        [
            "meeting_key",
            "session_key",
            "event_type",
            "start_lap",
            "end_lap",
            "event_time_utc",
            "category",
            "flag",
            "scope",
            "sector",
            "driver_number",
            "message",
        ]
    ].rename(
        columns={
            "event_type": "type",
        }
    )

    result = result.sort_values(
        ["event_time_utc", "start_lap"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 6. VALIDATE
    # --------------------------------------------------

    if result.empty:
        raise ValueError("No track events were extracted.")

    if result["event_time_utc"].isna().any():
        raise ValueError("Missing event timestamps found.")

    if result["start_lap"].isna().any():
        raise ValueError("Missing event start laps found.")

    if result["type"].isin(
        ["SC", "VSC", "RED", "YELLOW", "GREEN", "CLEAR"]
    ).all() is False:
        raise ValueError("Unexpected event types found.")

    logger.info("Track-event transformation completed.")
    logger.info("Total classified events: %s", len(result))

    logger.info(
        "Event distribution:\n%s",
        result["type"].value_counts().to_string(),
    )

    logger.info(
        "Safety car events: %s",
        result["type"].eq("SC").sum(),
    )

    logger.info(
        "VSC events: %s",
        result["type"].eq("VSC").sum(),
    )

    logger.info(
        "Red flag events: %s",
        result["type"].eq("RED").sum(),
    )

    logger.info(
        "Events with unknown end lap: %s",
        result["end_lap"].isna().sum(),
    )

    # --------------------------------------------------
    # 7. SAVE
    # --------------------------------------------------

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT_FILE,
        index=False,
        compression="snappy",
    )

    logger.info("Saved: %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
