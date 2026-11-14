from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "openf1" / "2024" / "bahrain_9472"

PROCESSED_FILE = (
    ROOT / "data" / "processed" / "bahrain_2024" / "laps.parquet"
)


def safe_text(value) -> str:
    """Convert a nullable value into uppercase text."""
    if pd.isna(value):
        return ""
    return str(value).upper().strip()



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



def main():
    race_control = pd.read_parquet(
        RAW_DIR / "race_control.parquet"
    )

    laps = pd.read_parquet(PROCESSED_FILE)

    race_control["classified_status"] = (
        race_control.apply(classify_event, axis=1)
    )

    print("\n1. ORIGINAL FLAG VALUES")
    print(
        race_control["flag"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n2. RACE CONTROL RECORDS CLASSIFIED AS RED")

    red_events = race_control[
        race_control["classified_status"] == "RED"
    ]

    if red_events.empty:
        print("No red events found.")
    else:
        print(
            red_events[
                ["date", "lap_number", "category", "flag", "message"]
            ].to_string(index=False)
        )

    print("\n3. ALL CLASSIFIED EVENTS")

    classified = race_control.dropna(
        subset=["classified_status"]
    )

    print(
        classified[
            ["date", "lap_number", "flag", "message",
             "classified_status"]
        ].to_string(index=False)
    )

    print("\n4. TRANSFORMED TRACK STATUS")

    print(
        laps["track_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n5. TRANSFORMED RED LAPS")

    red_laps = laps[
        laps["track_status"] == "RED"
    ]

    if red_laps.empty:
        print("No transformed red laps.")
    else:
        print(
            red_laps[
                ["driver", "lap", "track_status", "is_clean"]
            ].to_string(index=False)
        )

    print("\n6. PIT FLAG VALIDATION")

    print("Pit-in:", int(laps["pit_in"].sum()))
    print("Pit-out:", int(laps["pit_out"].sum()))

    both = laps[laps["pit_in"] & laps["pit_out"]]

    print("Both pit-in and pit-out on same lap:", len(both))

    print("\n7. FLAGGED LAP COUNTS BY LAP")

    print(
        laps.dropna(subset=["track_status"])
        .groupby(["lap", "track_status"])
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
