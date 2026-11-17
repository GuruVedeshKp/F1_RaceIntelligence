from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ARCHIVE = Path("data/raw/f1archive/f1archive-export-6fa7f631-parquet.zip")
REFERENCE_SESSIONS = [
    "2018-chinese-race",
    "2024-singapore-race",
    "2021-british-race",
    "2021-abu-dhabi-race",
    "2024-sao-paulo-race",
]
RED_FLAG_PATTERN = re.compile(
    r"\bRED FLAG\b|RACE SUSPENDED|SESSION SUSPENDED|SESSION STOPPED",
    flags=re.IGNORECASE,
)


def read_table(zf: zipfile.ZipFile, category: str, required: bool = True) -> pd.DataFrame:
    suffix = f"/{category}.parquet"
    matches = [
        name
        for name in zf.namelist()
        if name.lower().endswith(suffix) or name.lower() == f"{category}.parquet"
    ]
    if not matches:
        if required:
            raise RuntimeError(f"Missing {category}.parquet in archive.")
        return pd.DataFrame()
    if len(matches) > 1:
        raise RuntimeError(f"Multiple {category}.parquet matches: {matches}")
    with zf.open(matches[0]) as f:
        return pd.read_parquet(io.BytesIO(f.read()))


def numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def clean_compound(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    val = str(value).strip().upper()
    if val in {"", "UNKNOWN", "NONE", "NAN", "NULL"}:
        return None
    return val


def as_list(series: pd.Series) -> list[str]:
    vals = []
    for value in series.dropna().tolist():
        if isinstance(value, bool):
            vals.append(str(value))
        else:
            vals.append(str(value))
    return sorted(set(vals))


def fmt(value: Any, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}"
    return str(value)


def map_timestamp_to_lap(ts_ns: Any, driver_laps: pd.DataFrame) -> float | None:
    if pd.isna(ts_ns) or driver_laps.empty:
        return None
    if not {"lap_start_ns", "lap_end_ns", "lap"}.issubset(driver_laps.columns):
        return None
    inside = driver_laps[
        driver_laps["lap_start_ns"].notna()
        & driver_laps["lap_end_ns"].notna()
        & (driver_laps["lap_start_ns"] <= ts_ns)
        & (ts_ns <= driver_laps["lap_end_ns"])
    ]
    if inside.empty:
        return None
    return float(inside.iloc[0]["lap"])


def find_red_flag_rows(rc: pd.DataFrame, session: str) -> pd.DataFrame:
    if rc.empty:
        return pd.DataFrame()
    rows = []
    text_cols = [
        c for c in rc.columns
        if rc[c].dtype == object or pd.api.types.is_string_dtype(rc[c].dtype)
    ]
    for _, row in rc.iterrows():
        text = " | ".join(str(row[c]) for c in text_cols if pd.notna(row[c]))
        if RED_FLAG_PATTERN.search(text):
            record = {"session_key": session}
            for col in rc.columns:
                value = row[col]
                if isinstance(value, (str, int, float, bool)) or pd.isna(value):
                    record[col] = value
                else:
                    record[col] = str(value)
            record["matched_text"] = text[:1000]
            rows.append(record)
    return pd.DataFrame(rows)


def unique_compounds(group: pd.DataFrame) -> list[str]:
    if "compound" not in group.columns:
        return []
    return sorted({c for c in (clean_compound(x) for x in group["compound"]) if c})


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only multi-race audit of F1Archive tyre records grouped by driver and "
            "source stint number. Pit-out markers are candidates, not assumed tyre changes."
        )
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--sessions",
        default=",".join(REFERENCE_SESSIONS),
        help="Comma-separated session keys. Defaults to the five reference races.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cache/f1archive/stint_reconstruction_audit"),
    )
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    output_dir = args.output.resolve()
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]

    if not archive_path.is_file():
        print(f"ERROR: archive not found: {archive_path}", file=sys.stderr)
        return 2
    if not sessions:
        print("ERROR: no sessions supplied.", file=sys.stderr)
        return 2

    try:
        with zipfile.ZipFile(archive_path) as zf:
            stints_all = read_table(zf, "stints")
            laps_all = read_table(zf, "laps")
            pits_all = read_table(zf, "pit_stops")
            results_all = read_table(zf, "results")
            race_control_all = read_table(zf, "race_control", required=False)
    except Exception as exc:
        print(f"ERROR reading archive: {exc}", file=sys.stderr)
        return 2

    required = {
        "stints": {"session_key", "driver_number", "ts_ns", "stint", "compound"},
        "laps": {"session_key", "driver_number", "lap", "pit_out_ns", "pit_in_ns"},
        "pit_stops": {"session_key", "driver_id", "lap", "stop_number"},
        "results": {"session_key", "driver_id", "driver_number"},
    }
    for name, frame in [
        ("stints", stints_all),
        ("laps", laps_all),
        ("pit_stops", pits_all),
        ("results", results_all),
    ]:
        missing = required[name] - set(frame.columns)
        if missing:
            print(f"ERROR: {name}.parquet is missing columns: {sorted(missing)}", file=sys.stderr)
            return 2

    # Session-local driver-number mapping for the Jolpica-keyed pit-stop table.
    # Ignore missing/blank mapping keys, and collapse repeated rows only when the
    # same (session_key, driver_id) maps to one unambiguous driver_number.
    for frame_name, frame in (("results", results_all), ("pit_stops", pits_all)):
        frame["session_key"] = frame["session_key"].astype("string").str.strip()
        frame["driver_id"] = frame["driver_id"].astype("string").str.strip()
        frame.loc[frame["session_key"].eq(""), "session_key"] = pd.NA
        frame.loc[frame["driver_id"].eq(""), "driver_id"] = pd.NA

    numeric(results_all, ["driver_number"])
    mapping_candidates = results_all[
        ["session_key", "driver_id", "driver_number"]
    ].dropna(subset=["session_key", "driver_id", "driver_number"]).copy()

    mapping_conflicts = (
        mapping_candidates.groupby(["session_key", "driver_id"])["driver_number"]
        .nunique()
        .reset_index(name="driver_number_count")
        .query("driver_number_count > 1")
    )
    if not mapping_conflicts.empty:
        conflict_details = mapping_candidates.merge(
            mapping_conflicts[["session_key", "driver_id"]],
            on=["session_key", "driver_id"],
            how="inner",
        ).sort_values(["session_key", "driver_id", "driver_number"])
        print(
            "ERROR: some session/driver_id keys map to multiple car numbers. "
            "Refusing to guess. Conflicts:",
            file=sys.stderr,
        )
        print(conflict_details.to_string(index=False), file=sys.stderr)
        return 2

    driver_map = mapping_candidates.drop_duplicates(
        subset=["session_key", "driver_id"]
    )

    try:
        pits_all = pits_all.merge(
            driver_map,
            on=["session_key", "driver_id"],
            how="left",
            validate="many_to_one",
            indicator="_driver_map_status",
        )
    except Exception as exc:
        print(f"ERROR mapping pit-stop driver IDs to numbers: {exc}", file=sys.stderr)
        return 2

    unmapped_pits = pits_all[pits_all["driver_number"].isna()]
    if not unmapped_pits.empty:
        print(
            f"WARNING: {len(unmapped_pits)} pit-stop rows could not be mapped to a driver number. "
            "They will be excluded from per-driver pit-stop counts.",
            file=sys.stderr,
        )
        print(
            unmapped_pits[["session_key", "driver_id", "lap", "stop_number"]]
            .head(20).to_string(index=False),
            file=sys.stderr,
        )
    pits_all = pits_all.drop(columns=["_driver_map_status"])

    for frame in (stints_all, laps_all, pits_all, results_all, race_control_all):
        if not frame.empty and "driver_number" in frame.columns:
            frame["driver_number"] = pd.to_numeric(frame["driver_number"], errors="coerce")

    numeric(stints_all, ["ts_ns", "stint", "stint_age", "start_lap", "max_age", "lap_flags"])
    numeric(laps_all, ["lap", "lap_start_ns", "lap_end_ns", "pit_in_ns", "pit_out_ns"])
    numeric(pits_all, ["lap", "stop_number", "driver_number"])

    stint_rows: list[dict[str, Any]] = []
    driver_rows: list[dict[str, Any]] = []
    session_rows: list[dict[str, Any]] = []
    same_timestamp_rows: list[dict[str, Any]] = []
    pit_out_rows: list[dict[str, Any]] = []
    red_flag_frames: list[pd.DataFrame] = []
    report: list[str] = []

    report.extend([
        "F1ARCHIVE STINT-NUMBER RECONSTRUCTION AUDIT",
        "=" * 120,
        f"Archive: {archive_path}",
        f"Sessions: {', '.join(sessions)}",
        "",
        "Interpretation rules:",
        "- Source stint IDs are grouped by session and driver; no timestamp-nearest tyre assignment is made.",
        "- A pit-out is counted as a candidate boundary only; it is not assumed to mean a tyre change.",
        "- Different stint IDs at one timestamp are reported as a possible old/new-state transition.",
        "- Multiple compounds within one driver/stint ID are flagged for review, not resolved automatically.",
        "- Source tyre age is preserved as observed; it is not treated as race-only age.",
        "- Red flags are contextual evidence, not automatic tyre-stint increments.",
        "",
    ])

    for session in sessions:
        ss = stints_all[stints_all["session_key"] == session].copy()
        sl = laps_all[laps_all["session_key"] == session].copy()
        sp = pits_all[pits_all["session_key"] == session].copy()
        sr = results_all[results_all["session_key"] == session].copy()
        src = (
            race_control_all[race_control_all["session_key"] == session].copy()
            if not race_control_all.empty and "session_key" in race_control_all.columns
            else pd.DataFrame()
        )

        if ss.empty and sl.empty:
            report.append(f"SESSION {session}: NOT FOUND in stints/laps tables")
            session_rows.append({
                "session_key": session,
                "session_found": False,
                "stint_rows": 0,
                "lap_rows": 0,
                "pit_stop_rows": 0,
                "red_flag_messages": 0,
                "drivers": 0,
            })
            continue

        red_flags = find_red_flag_rows(src, session)
        if not red_flags.empty:
            red_flag_frames.append(red_flags)

        session_driver_numbers = sorted({
            int(x) for x in pd.concat([
                ss.get("driver_number", pd.Series(dtype=float)),
                sl.get("driver_number", pd.Series(dtype=float)),
                sp.get("driver_number", pd.Series(dtype=float)),
            ]).dropna().unique()
        })

        report.extend([
            "",
            "-" * 120,
            f"SESSION: {session}",
            f"Stint rows={len(ss)} | lap rows={len(sl)} | pit-stop rows={len(sp)} | "
            f"red-flag messages={len(red_flags)} | drivers={len(session_driver_numbers)}",
        ])

        session_driver_count = 0
        session_compound_conflict_stints = 0
        session_same_ts_multi_stint = 0
        session_same_ts_same_stint_compound_conflicts = 0
        session_pit_outs = 0
        session_pit_stops = len(sp)

        for driver in session_driver_numbers:
            ds = ss[ss["driver_number"] == driver].copy()
            dl = sl[sl["driver_number"] == driver].copy().sort_values("lap")
            dp = sp[sp["driver_number"] == driver].copy().sort_values("lap")
            if ds.empty and dl.empty and dp.empty:
                continue

            session_driver_count += 1

            # Pit-out rows are on the out-lap in this export. Preserve the actual marker.
            outs = dl[dl["pit_out_ns"].notna()].copy()
            session_pit_outs += len(outs)
            for _, out in outs.iterrows():
                pit_out_rows.append({
                    "session_key": session,
                    "driver_number": driver,
                    "out_lap": out.get("lap"),
                    "pit_out_ns": out.get("pit_out_ns"),
                    "pit_out_seconds": (
                        float(out["pit_out_ns"]) / 1e9
                        if pd.notna(out.get("pit_out_ns")) else None
                    ),
                })

            # Driver-level source stint order and monotonicity.
            ds = ds.sort_values(["ts_ns", "stint"], kind="stable").reset_index(drop=True)
            stint_ids = sorted({
                int(x) for x in ds["stint"].dropna().unique()
            })
            stint_backsteps = 0
            stint_sequence = ds["stint"].dropna().tolist()
            for prev, current in zip(stint_sequence, stint_sequence[1:]):
                if current < prev:
                    stint_backsteps += 1

            compound_conflicts = 0
            driver_stint_count = 0
            driver_age_decreases = 0

            for stint_id, group in ds.groupby("stint", dropna=False, sort=True):
                if pd.isna(stint_id):
                    continue
                driver_stint_count += 1
                group = group.sort_values("ts_ns", kind="stable").copy()
                compounds = unique_compounds(group)
                conflict = len(compounds) > 1
                if conflict:
                    compound_conflicts += 1
                    session_compound_conflict_stints += 1

                ages = pd.to_numeric(group.get("stint_age"), errors="coerce").dropna()
                age_decreases = int((ages.diff() < 0).sum()) if len(ages) > 1 else 0
                driver_age_decreases += age_decreases

                first_ts = group["ts_ns"].dropna().min() if group["ts_ns"].notna().any() else None
                last_ts = group["ts_ns"].dropna().max() if group["ts_ns"].notna().any() else None
                first_lap = map_timestamp_to_lap(first_ts, dl) if first_ts is not None else None
                last_lap = map_timestamp_to_lap(last_ts, dl) if last_ts is not None else None

                record = {
                    "session_key": session,
                    "driver_number": driver,
                    "stint": int(stint_id),
                    "rows": len(group),
                    "compounds": "|".join(compounds),
                    "distinct_compound_count": len(compounds),
                    "same_stint_compound_conflict": conflict,
                    "first_seen_ns": first_ts,
                    "last_seen_ns": last_ts,
                    "first_seen_lap": first_lap,
                    "last_seen_lap": last_lap,
                    "stint_age_min": ages.min() if not ages.empty else None,
                    "stint_age_max": ages.max() if not ages.empty else None,
                    "stint_age_decreases": age_decreases,
                    "new_values": "|".join(as_list(group["new"])) if "new" in group.columns else "",
                    "tyres_changed_values": (
                        "|".join(as_list(group["tyres_changed"]))
                        if "tyres_changed" in group.columns else ""
                    ),
                    "start_lap_values": (
                        "|".join(as_list(group["start_lap"]))
                        if "start_lap" in group.columns else ""
                    ),
                    "lap_flags_values": (
                        "|".join(as_list(group["lap_flags"]))
                        if "lap_flags" in group.columns else ""
                    ),
                }
                stint_rows.append(record)

                report.append(
                    f"  Driver #{driver:>2} | stint {int(stint_id):>2} | "
                    f"rows={len(group):>3} | compounds={','.join(compounds) or '-'} | "
                    f"age={fmt(ages.min() if not ages.empty else None, 0)}.."
                    f"{fmt(ages.max() if not ages.empty else None, 0)} | "
                    f"age_decreases={age_decreases} | "
                    f"first_lap={fmt(first_lap, 0)} | last_lap={fmt(last_lap, 0)}"
                    + (" | SAME_STINT_COMPOUND_CONFLICT" if conflict else "")
                )

            # Same timestamp groups: separate cross-stint hand-offs from same-stint conflicts.
            driver_multi_stint_ts = 0
            driver_same_stint_multi_comp_ts = 0
            for ts_ns, group in ds.groupby("ts_ns", dropna=False, sort=True):
                if pd.isna(ts_ns) or len(group) < 2:
                    continue
                n_stints = group["stint"].nunique(dropna=True)
                compound_values = unique_compounds(group)
                same_stint_compounds = []
                for sid, same_stint_group in group.groupby("stint", dropna=False):
                    comps = unique_compounds(same_stint_group)
                    if len(comps) > 1:
                        same_stint_compounds.append(f"{sid}:{'/'.join(comps)}")
                if n_stints > 1:
                    driver_multi_stint_ts += 1
                    session_same_ts_multi_stint += 1
                if same_stint_compounds:
                    driver_same_stint_multi_comp_ts += 1
                    session_same_ts_same_stint_compound_conflicts += 1

                same_timestamp_rows.append({
                    "session_key": session,
                    "driver_number": driver,
                    "ts_ns": ts_ns,
                    "ts_seconds": float(ts_ns) / 1e9,
                    "rows": len(group),
                    "stint_ids": "|".join(as_list(group["stint"])),
                    "compounds": "|".join(compound_values),
                    "same_stint_compound_conflicts": "|".join(same_stint_compounds),
                    "classification": (
                        "SAME_STINT_MULTIPLE_COMPOUNDS" if same_stint_compounds
                        else "MULTIPLE_STINT_IDS_POSSIBLE_HANDOFF" if n_stints > 1
                        else "DUPLICATE_OR_REPEAT_UPDATE"
                    ),
                })

            # Pit-stop ↔ pit-out comparison. Pit lap N is expected to have an out-lap
            # marker on N+1 for the observed archive convention; this is diagnostic only.
            missing_expected_outlap = 0
            for _, pit in dp.iterrows():
                pit_lap = pit.get("lap")
                expected_out_lap = pit_lap + 1 if pd.notna(pit_lap) else None
                matches = (
                    outs[outs["lap"] == expected_out_lap]
                    if expected_out_lap is not None else pd.DataFrame()
                )
                has_expected_out = not matches.empty
                if not has_expected_out:
                    missing_expected_outlap += 1
                pit_out_rows.append({
                    "session_key": session,
                    "driver_number": driver,
                    "pit_stop_number": pit.get("stop_number"),
                    "pit_lap": pit_lap,
                    "expected_out_lap": expected_out_lap,
                    "expected_out_lap_marker_found": has_expected_out,
                    "candidate_out_lap_markers_for_driver": "|".join(
                        str(int(x)) for x in outs["lap"].dropna().tolist()
                    ),
                })

            # Candidate segment count based on explicit out-lap markers only.
            # It is deliberately not labelled as verified tyre-stint count.
            candidate_segments = len(outs) + 1
            expected_from_pits = len(dp) + 1

            report.append(
                f"  Driver #{driver:>2} SUMMARY | pit_stops={len(dp)} | "
                f"pit_out_markers={len(outs)} | candidate_segments_from_outs={candidate_segments} | "
                f"source_stint_ids={stint_ids} | source_stint_id_count={len(stint_ids)} | "
                f"pit_stops_plus_one={expected_from_pits} | "
                f"missing_expected_outlap_markers={missing_expected_outlap} | "
                f"stint_backsteps={stint_backsteps} | "
                f"same_ts_multi_stint_groups={driver_multi_stint_ts} | "
                f"same_ts_same_stint_compound_conflicts={driver_same_stint_multi_comp_ts} | "
                f"same_stint_compound_conflicts={compound_conflicts} | "
                f"stint_age_decreases={driver_age_decreases}"
            )

            driver_rows.append({
                "session_key": session,
                "driver_number": driver,
                "pit_stop_count": len(dp),
                "pit_out_marker_count": len(outs),
                "candidate_segments_from_pit_out_markers": candidate_segments,
                "pit_stops_plus_one_diagnostic": expected_from_pits,
                "source_stint_ids": "|".join(str(x) for x in stint_ids),
                "source_stint_id_count": len(stint_ids),
                "same_stint_compound_conflict_count": compound_conflicts,
                "same_timestamp_multi_stint_groups": driver_multi_stint_ts,
                "same_timestamp_same_stint_compound_conflicts": driver_same_stint_multi_comp_ts,
                "stint_age_decreases": driver_age_decreases,
                "stint_id_backsteps": stint_backsteps,
                "pit_stops_without_expected_outlap_marker": missing_expected_outlap,
            })

        session_rows.append({
            "session_key": session,
            "session_found": True,
            "stint_rows": len(ss),
            "lap_rows": len(sl),
            "pit_stop_rows": len(sp),
            "red_flag_messages": len(red_flags),
            "drivers": session_driver_count,
            "pit_out_markers": session_pit_outs,
            "same_timestamp_multi_stint_groups": session_same_ts_multi_stint,
            "same_timestamp_same_stint_compound_conflicts": session_same_ts_same_stint_compound_conflicts,
            "same_stint_compound_conflicts": session_compound_conflict_stints,
        })

        if not red_flags.empty:
            report.append("  RED-FLAG / SUSPENSION MESSAGES:")
            for _, rr in red_flags.iterrows():
                report.append(f"    {rr.get('matched_text', '')[:350]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(session_rows).to_csv(output_dir / "session_summary.csv", index=False)
    pd.DataFrame(driver_rows).to_csv(output_dir / "driver_summary.csv", index=False)
    pd.DataFrame(stint_rows).to_csv(output_dir / "driver_stint_summary.csv", index=False)
    pd.DataFrame(same_timestamp_rows).to_csv(
        output_dir / "same_timestamp_state_groups.csv", index=False
    )
    pd.DataFrame(pit_out_rows).to_csv(output_dir / "pit_out_and_stop_markers.csv", index=False)
    if red_flag_frames:
        pd.concat(red_flag_frames, ignore_index=True).to_csv(
            output_dir / "race_control_red_flag_messages.csv", index=False
        )
    else:
        pd.DataFrame(columns=["session_key", "matched_text"]).to_csv(
            output_dir / "race_control_red_flag_messages.csv", index=False
        )

    report.extend([
        "",
        "=" * 120,
        "SUMMARY",
        f"Reference sessions requested: {len(sessions)}",
        f"Sessions found: {sum(1 for r in session_rows if r.get('session_found'))}",
        f"Driver rows: {len(driver_rows)}",
        f"Driver/stint groups: {len(stint_rows)}",
        f"Same-timestamp groups: {len(same_timestamp_rows)}",
        "",
        "Output files:",
        f"  {output_dir / 'session_summary.csv'}",
        f"  {output_dir / 'driver_summary.csv'}",
        f"  {output_dir / 'driver_stint_summary.csv'}",
        f"  {output_dir / 'same_timestamp_state_groups.csv'}",
        f"  {output_dir / 'pit_out_and_stop_markers.csv'}",
        f"  {output_dir / 'race_control_red_flag_messages.csv'}",
        "",
        "No tyre state was reconstructed or overwritten by this audit.",
        "No source archive, raw JSON, cache, or PostgreSQL record was modified.",
    ])
    report_path = output_dir / "stint_reconstruction_audit.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("F1ARCHIVE STINT-NUMBER RECONSTRUCTION AUDIT")
    print("=" * 100)
    print(f"Archive: {archive_path}")
    print(f"Sessions requested: {len(sessions)}")
    print(f"Sessions found: {sum(1 for r in session_rows if r.get('session_found'))}")
    print(f"Driver/stint groups: {len(stint_rows)}")
    print(f"Same-timestamp groups: {len(same_timestamp_rows)}")
    print(f"Output: {output_dir}")
    print(f"Main report: {report_path}")
    print("\nSESSION SUMMARY")
    if session_rows:
        print(pd.DataFrame(session_rows).to_string(index=False))
    print("\nNo source archive, raw JSON, cache, or PostgreSQL records were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
