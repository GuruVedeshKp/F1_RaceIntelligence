from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_URL = "https://api.openf1.org/v1"

YEAR = 2024
MEETING_KEY = 1229
SESSION_KEY = 9472

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "openf1"
    / str(YEAR)
    / f"bahrain_{SESSION_KEY}"
)

# Conservative pacing for the free API tier.
REQUEST_DELAY_SECONDS = 2.1
MAX_RETRIES = 4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# ENDPOINT CONFIGURATION
# --------------------------------------------------

ENDPOINTS: dict[str, dict[str, int]] = {
    "sessions": {
        "session_key": SESSION_KEY,
    },
    "drivers": {
        "session_key": SESSION_KEY,
    },
    "laps": {
        "session_key": SESSION_KEY,
    },
    "stints": {
        "session_key": SESSION_KEY,
    },
    "pit": {
        "session_key": SESSION_KEY,
    },
    "position": {
        "session_key": SESSION_KEY,
    },
    "intervals": {
        "session_key": SESSION_KEY,
    },
    "race_control": {
        "session_key": SESSION_KEY,
    },
    "weather": {
        "session_key": SESSION_KEY,
    },
    "session_result": {
        "session_key": SESSION_KEY,
    },
}


# --------------------------------------------------
# API REQUEST FUNCTION
# --------------------------------------------------

def fetch_endpoint(
    client: httpx.Client,
    endpoint: str,
    params: dict[str, int],
) -> list[dict[str, Any]]:
    """Fetch one OpenF1 endpoint with basic retry handling."""

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = client.get(
                f"/{endpoint}",
                params=params,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                try:
                    wait_time = float(retry_after) if retry_after else 10.0
                except ValueError:
                    wait_time = 10.0

                logger.warning(
                    "Rate limited on %s. Waiting %.1f seconds.",
                    endpoint,
                    wait_time,
                )

                if attempt == MAX_RETRIES:
                    response.raise_for_status()

                time.sleep(wait_time)
                continue

            if response.status_code in (500, 502, 503, 504):
                response.raise_for_status()

            response.raise_for_status()

            records = response.json()

            if not isinstance(records, list):
                raise TypeError(
                    f"Unexpected response format from {endpoint}"
                )

            return records

        except httpx.HTTPStatusError as exc:

            if (
                exc.response.status_code in (500, 502, 503, 504)
                and attempt < MAX_RETRIES
            ):
                wait_time = min(2 ** attempt, 30)

                logger.warning(
                    "%s returned HTTP %s. Retrying in %s seconds.",
                    endpoint,
                    exc.response.status_code,
                    wait_time,
                )

                time.sleep(wait_time)
                continue

            raise

        except httpx.RequestError:

            if attempt == MAX_RETRIES:
                raise

            wait_time = min(2 ** attempt, 30)

            logger.warning(
                "Connection error on %s. Retrying in %s seconds.",
                endpoint,
                wait_time,
            )

            time.sleep(wait_time)

    raise RuntimeError(f"Could not retrieve endpoint: {endpoint}")


# --------------------------------------------------
# SAVE RAW JSON AND PARQUET
# --------------------------------------------------

def save_endpoint_data(
    endpoint: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Save original JSON and a Parquet-compatible tabular copy."""

    json_path = OUTPUT_DIR / f"{endpoint}.json"
    parquet_path = OUTPUT_DIR / f"{endpoint}.parquet"

    def is_missing(value: Any) -> bool:
        """Safely identify missing scalar values."""
        if value is None:
            return True

        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    # 1. Preserve the original API response without changing its values.
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            records,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # 2. Convert records to a DataFrame.
    dataframe = pd.DataFrame.from_records(records)

    if not dataframe.empty:

        # 3. Serialize nested dictionaries and lists.
        for column in dataframe.columns:

            dataframe[column] = dataframe[column].map(
                lambda value: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                )
            )

        # 4. Detect columns containing multiple Python types.
        # Example: gap_to_leader = 0.452 or "+1 LAP".
        for column in dataframe.columns:

            if dataframe[column].dtype == "object":

                non_missing_values = [
                    value
                    for value in dataframe[column]
                    if not is_missing(value)
                ]

                value_types = {
                    type(value)
                    for value in non_missing_values
                }

                if len(value_types) > 1:

                    logger.warning(
                        "Mixed types detected in '%s': %s. "
                        "Converting raw Parquet column to string.",
                        column,
                        [value_type.__name__ for value_type in value_types],
                    )

                    dataframe[column] = dataframe[column].map(
                        lambda value: (
                            None
                            if is_missing(value)
                            else str(value)
                        )
                    )

        # 5. Save Parquet.
        dataframe.to_parquet(
            parquet_path,
            index=False,
            engine="pyarrow",
            compression="snappy",
        )

        logger.info(
            "%s: %s records | Parquet saved",
            endpoint,
            len(dataframe),
        )

    else:
        logger.warning(
            "%s returned no records. Empty JSON response saved.",
            endpoint,
        )

    return {
        "record_count": len(records),
        "json_file": str(json_path.relative_to(PROJECT_ROOT)),
        "parquet_file": (
            str(parquet_path.relative_to(PROJECT_ROOT))
            if not dataframe.empty
            else None
        ),
    }



# --------------------------------------------------
# MAIN INGESTION PIPELINE
# --------------------------------------------------

def main() -> None:
    """Download all selected Bahrain Grand Prix datasets."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info("Starting OpenF1 ingestion.")
    logger.info("Race: 2024 Bahrain Grand Prix")
    logger.info("Session key: %s", SESSION_KEY)
    logger.info("Output: %s", OUTPUT_DIR)

    ingestion_results: dict[str, Any] = {}

    with httpx.Client(
        base_url=BASE_URL,
        timeout=60.0,
        headers={
            "User-Agent": "F1-Race-Intelligence/0.1",
        },
    ) as client:

        for index, (endpoint, params) in enumerate(
            ENDPOINTS.items()
        ):

            logger.info(
                "Fetching %s (%s/%s)",
                endpoint,
                index + 1,
                len(ENDPOINTS),
            )

            records = fetch_endpoint(
                client,
                endpoint,
                params,
            )

            ingestion_results[endpoint] = save_endpoint_data(
                endpoint,
                records,
            )

            # Stay within the conservative request rate.
            if index < len(ENDPOINTS) - 1:
                time.sleep(REQUEST_DELAY_SECONDS)

    # Create an ingestion manifest.
    manifest = {
        "source": "OpenF1 API",
        "year": YEAR,
        "grand_prix": "Bahrain Grand Prix",
        "circuit": "Sakhir",
        "meeting_key": MEETING_KEY,
        "session_key": SESSION_KEY,
        "retrieved_at_utc": datetime.now(
            UTC
        ).isoformat(),
        "endpoints": ingestion_results,
    }

    manifest_path = OUTPUT_DIR / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    total_records = sum(
        item["record_count"]
        for item in ingestion_results.values()
    )

    logger.info("=" * 55)
    logger.info("INGESTION COMPLETED")
    logger.info("Total records retrieved: %s", total_records)
    logger.info("Manifest: %s", manifest_path)
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
