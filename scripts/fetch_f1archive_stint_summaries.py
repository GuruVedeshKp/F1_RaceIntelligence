from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The project archive is expected at this path relative to the project root.
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "f1archive" / "f1archive-export-6fa7f631-parquet.zip"
API_BASE = "https://f1archive.org/api/v1"
USER_AGENT = "F1-Race-Intelligence/0.1 (local historical data audit)"


def find_zip_member(zf: ZipFile, basename: str) -> str:
    matches = [name for name in zf.namelist() if Path(name).name.lower() == basename.lower()]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {basename!r} inside the export ZIP; found {len(matches)}. "
            f"Matching members: {matches[:10]}"
        )
    return matches[0]


def load_race_sessions(archive_path: Path) -> pd.DataFrame:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Export ZIP not found: {archive_path}")

    try:
        with ZipFile(archive_path) as zf:
            member = find_zip_member(zf, "sessions.parquet")
            sessions = pd.read_parquet(BytesIO(zf.read(member)))
    except BadZipFile as exc:
        raise RuntimeError(f"The archive is not a valid ZIP file: {archive_path}") from exc

    if "session_key" not in sessions.columns:
        raise RuntimeError(
            "sessions.parquet has no 'session_key' column. "
            f"Available columns: {list(sessions.columns)}"
        )

    parsed = sessions["session_key"].astype(str).str.extract(
        r"^(?P<year>\d{4})-(?P<event>.+)-race$"
    )
    sessions = sessions.copy()
    sessions["year_num"] = pd.to_numeric(parsed["year"], errors="coerce")
    sessions["event_slug"] = parsed["event"]

    sessions = sessions.loc[
        sessions["year_num"].between(2018, 2025) & sessions["event_slug"].notna()
    ].copy()

    # If session_kind exists, use it as an additional guard against non-race rows.
    if "session_kind" in sessions.columns:
        sessions = sessions.loc[
            sessions["session_kind"].astype(str).str.lower().eq("race")
        ].copy()

    sort_columns = ["year_num"]
    if "round" in sessions.columns:
        sessions["round_sort"] = pd.to_numeric(sessions["round"], errors="coerce")
        sort_columns.append("round_sort")
    sort_columns.append("session_key")
    sessions = sessions.sort_values(sort_columns).reset_index(drop=True)

    if sessions["session_key"].duplicated().any():
        duplicates = sessions.loc[sessions["session_key"].duplicated(), "session_key"].tolist()
        raise RuntimeError(f"Duplicate race session keys found: {duplicates[:10]}")

    return sessions


def count_records(payload: object) -> int | None:
    """Best-effort count for common F1Archive response shapes."""
    if isinstance(payload, list):
        return len(payload)
    if not isinstance(payload, dict):
        return None

    for key in ("stints", "data", "rows", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)

    drivers = payload.get("drivers")
    if isinstance(drivers, dict):
        total = 0
        found_list = False
        for value in drivers.values():
            if isinstance(value, list):
                total += len(value)
                found_list = True
            elif isinstance(value, dict):
                for nested_key in ("stints", "data", "rows", "items"):
                    nested = value.get(nested_key)
                    if isinstance(nested, list):
                        total += len(nested)
                        found_list = True
                        break
        if found_list:
            return total

    return None


def request_json(url: str, timeout: int = 45, max_retries: int = 5) -> tuple[object, dict[str, str]]:
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }

    for attempt in range(max_retries + 1):
        request = Request(url, headers=headers, method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
                response_headers = {k.lower(): v for k, v in response.headers.items()}
                return json.loads(body.decode("utf-8")), response_headers

        except HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                retry_after = exc.headers.get("Retry-After")
                try:
                    wait_seconds = max(5.0, float(retry_after)) if retry_after else 60.0
                except ValueError:
                    wait_seconds = 60.0
                print(f"  Rate limited (HTTP 429). Waiting {wait_seconds:.0f}s before retry...")
                time.sleep(wait_seconds)
                continue

            if exc.code >= 500 and attempt < max_retries:
                wait_seconds = min(60.0, 2.0 ** (attempt + 1))
                print(f"  Server error HTTP {exc.code}. Retrying in {wait_seconds:.0f}s...")
                time.sleep(wait_seconds)
                continue

            raise

        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt >= max_retries:
                raise
            wait_seconds = min(60.0, 2.0 ** (attempt + 1))
            print(f"  Request/response error: {exc}. Retrying in {wait_seconds:.0f}s...")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Request failed after retries: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and locally cache F1Archive dedicated race tyre-stint summaries."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to the previously downloaded F1Archive Parquet ZIP.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds between uncached API requests (default: 1.5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of races to process (useful for a small test run).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch summaries even when a local cached JSON file already exists.",
    )
    args = parser.parse_args()

    if args.delay < 0:
        parser.error("--delay must be zero or greater")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    sessions = load_race_sessions(args.archive)
    if args.limit is not None:
        sessions = sessions.head(args.limit).copy()

    output_dir = PROJECT_ROOT / "data" / "raw" / "f1archive" / "stint_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT_ROOT / "data" / "cache" / "f1archive" / "stint_summary_fetch_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print("F1ARCHIVE DEDICATED STINT SUMMARY FETCH")
    print("=" * 78)
    print(f"Archive:        {args.archive}")
    print(f"Race sessions:  {len(sessions)}")
    print(f"Cache folder:   {output_dir}")
    print(f"Request delay:  {args.delay:.1f}s")
    print(f"Refresh cache:  {args.refresh}")
    print()

    manifest_rows: list[dict[str, object]] = []
    request_count = 0
    cached_count = 0
    success_count = 0
    error_count = 0
    year_stats: dict[int, dict[str, int]] = {}

    for row in sessions.itertuples(index=False):
        session_key = str(getattr(row, "session_key"))
        year = int(getattr(row, "year_num"))
        event_slug = str(getattr(row, "event_slug"))
        cache_path = output_dir / str(year) / f"{event_slug}-race.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        year_stats.setdefault(year, {"success": 0, "cached": 0, "errors": 0})
        status = "error"
        record_count_value: int | None = None
        error_message = ""
        http_status: int | str = "cached"
        fetched_at_utc = ""

        if cache_path.is_file() and not args.refresh:
            try:
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                record_count_value = count_records(payload)
                status = "cached"
                cached_count += 1
                year_stats[year]["cached"] += 1
                fetched_at_utc = datetime.fromtimestamp(
                    cache_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            except (OSError, json.JSONDecodeError) as exc:
                error_message = f"Existing cache file could not be read: {exc}"
                status = "error"
                error_count += 1
                year_stats[year]["errors"] += 1
        else:
            if request_count > 0 and args.delay:
                time.sleep(args.delay)
            url = f"{API_BASE}/sessions/{year}/{event_slug}/race/stints"
            print(f"[{request_count + cached_count + error_count + 1:03d}/{len(sessions):03d}] {session_key}")
            try:
                payload, response_headers = request_json(url)
                cache_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                record_count_value = count_records(payload)
                status = "downloaded"
                http_status = 200
                fetched_at_utc = datetime.now(timezone.utc).isoformat()
                success_count += 1
                year_stats[year]["success"] += 1
                request_count += 1
                remaining = response_headers.get("x-ratelimit-remaining")
                reset = response_headers.get("x-ratelimit-reset")
                if remaining is not None:
                    print(f"    Cached {record_count_value} stint records; rate-limit remaining: {remaining}")
                else:
                    print(f"    Cached {record_count_value} stint records")
                if reset is not None:
                    print(f"    Rate-limit reset header: {reset}")
            except Exception as exc:  # Continue and record a failed session; reruns can retry it.
                status = "error"
                error_message = f"{type(exc).__name__}: {exc}"
                error_count += 1
                year_stats[year]["errors"] += 1
                request_count += 1
                print(f"    ERROR: {error_message}")

        manifest_rows.append(
            {
                "session_key": session_key,
                "year": year,
                "event_slug": event_slug,
                "status": status,
                "http_status": http_status,
                "stint_record_count": record_count_value,
                "cache_path": str(cache_path.relative_to(PROJECT_ROOT)),
                "fetched_at_utc": fetched_at_utc,
                "error": error_message,
            }
        )

        # Persist progress after each race so an interrupted run can resume safely.
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print()
    print("FETCH SUMMARY")
    print("=" * 78)
    print(f"Sessions processed: {len(sessions)}")
    print(f"New downloads:      {success_count}")
    print(f"Already cached:     {cached_count}")
    print(f"Errors:             {error_count}")
    print()
    print("BY SEASON")
    for year in sorted(year_stats):
        stats = year_stats[year]
        print(
            f"{year}: downloaded={stats['success']}, "
            f"cached={stats['cached']}, errors={stats['errors']}"
        )
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Raw JSON:  {output_dir}")
    if error_count:
        print("Some sessions failed. Re-run the same command; successful cache files will be skipped.")


if __name__ == "__main__":
    main()
