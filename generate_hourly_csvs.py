"""
generate_hourly_csvs.py
=======================

INPUT  : data/raw/Streaming_History_Audio_*.json   (Spotify extended history)
OUTPUT : data/viz/viz_hourly_intensity.csv
         data/viz/viz_hour_by_year.csv
         data/viz/viz_night_sessions.csv
         data/viz/viz_weekday_hourly.csv

HOW TO RUN
----------
  python generate_hourly_csvs.py

"""

import json
import csv
import glob
import os
from datetime import datetime, timezone

# ── CONFIG — adjust these to match Adam's folder structure ──────────────────
INPUT_GLOB  = "data/raw/Streaming_History_Audio_*.json"   # where raw JSON lives
OUTPUT_DIR  = "data/viz"                        # where existing CSVs live
MIN_MS      = 30_000    # ignore plays under 30 seconds (skips, accidental plays)
NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4}    # hours considered "late night"
# ────────────────────────────────────────────────────────────────────────────


def load_all_streams(pattern):
    """Load and merge all Streaming_History JSON files."""
    records = []
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files found matching: {pattern}\n"
            f"Check INPUT_GLOB at the top of this script."
        )
    print(f"Found {len(files)} JSON file(s):")
    for f in files:
        print(f"  {f}")
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
            records.extend(data)
    print(f"  → {len(records):,} total stream records loaded")
    return records


def parse_record(r):
    """
    Parse one Spotify extended streaming history record.
    Returns a dict with typed fields, or None if the record should be skipped.

    Spotify extended history fields used:
      ts                                  — ISO 8601 timestamp (UTC)
      ms_played                           — milliseconds played
      master_metadata_track_name          — track name
      master_metadata_album_artist_name   — artist name
      shuffle                             — bool
      skipped                             — bool or None
    """
    ms = r.get("ms_played", 0)
    if ms < MIN_MS:
        return None                     # skip very short plays

    ts_str = r.get("ts", "")
    if not ts_str:
        return None

    try:
        # Spotify exports UTC timestamps as "2021-03-15T23:45:00Z"
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None

    return {
        "ts":           ts,
        "year":         ts.year,
        "month":        ts.month,
        "year_month":   ts.strftime("%Y-%m"),
        "date":         ts.strftime("%Y-%m-%d"),
        "hour":         ts.hour,
        "dow":          ts.weekday(),       # 0=Mon … 6=Sun
        "is_weekend":   ts.weekday() >= 5,  # Sat=5, Sun=6
        "is_night":     ts.hour in NIGHT_HOURS,
        "ms_played":    ms,
        "hours_played": ms / 3_600_000,
        "artist":       r.get("master_metadata_album_artist_name") or "Unknown",
        "track":        r.get("master_metadata_track_name") or "Unknown",
        "shuffle":      bool(r.get("shuffle", False)),
        "skipped":      bool(r.get("skipped", False)),
    }


# ── OUTPUT 1 — viz_hourly_intensity.csv ─────────────────────────────────────
# One row per (year_month, hour).  Used for the radial clock and heatmap.
# Columns: year_month, hour, hours_played, track_count, is_night
def write_hourly_intensity(records, out_dir):
    from collections import defaultdict
    agg = defaultdict(lambda: {"hours_played": 0.0, "track_count": 0})

    for r in records:
        key = (r["year_month"], r["hour"])
        agg[key]["hours_played"] += r["hours_played"]
        agg[key]["track_count"]  += 1

    path = os.path.join(out_dir, "viz_hourly_intensity.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year_month", "hour", "hours_played", "track_count", "is_night"])
        for (ym, h), v in sorted(agg.items()):
            w.writerow([ym, h, round(v["hours_played"], 4),
                        v["track_count"], int(h in NIGHT_HOURS)])
    print(f"  ✓ {path}  ({len(agg):,} rows)")


# ── OUTPUT 2 — viz_hour_by_year.csv ─────────────────────────────────────────
# One row per (year, hour).  Aggregated across all months of that year.
# Used for the hour × year.
# Columns: year, hour, hours_played, track_count, is_night
def write_hour_by_year(records, out_dir):
    from collections import defaultdict
    agg = defaultdict(lambda: {"hours_played": 0.0, "track_count": 0})

    for r in records:
        key = (r["year"], r["hour"])
        agg[key]["hours_played"] += r["hours_played"]
        agg[key]["track_count"]  += 1

    path = os.path.join(out_dir, "viz_hour_by_year.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year", "hour", "hours_played", "track_count", "is_night"])
        for (yr, h), v in sorted(agg.items()):
            w.writerow([yr, h, round(v["hours_played"], 4),
                        v["track_count"], int(h in NIGHT_HOURS)])
    print(f"  ✓ {path}  ({len(agg):,} rows)")


# ── OUTPUT 3 — viz_night_sessions.csv ───────────────────────────────────────
# One row per month, night hours only (22h–04h). 
# Used for the night sessions summary chart.
# Columns: year_month, year, night_hours, night_tracks, total_hours,
#          night_share, top_night_artist
def write_night_sessions(records, out_dir):
    from collections import defaultdict

    # Monthly totals (all hours)
    monthly_total = defaultdict(float)
    # Monthly night totals
    monthly_night = defaultdict(lambda: {"hours": 0.0, "tracks": 0,
                                          "artists": defaultdict(float)})
    for r in records:
        ym = r["year_month"]
        monthly_total[ym] += r["hours_played"]
        if r["is_night"]:
            monthly_night[ym]["hours"]  += r["hours_played"]
            monthly_night[ym]["tracks"] += 1
            monthly_night[ym]["artists"][r["artist"]] += r["hours_played"]

    path = os.path.join(out_dir, "viz_night_sessions.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year_month", "year", "night_hours", "night_tracks",
                    "total_hours", "night_share", "top_night_artist"])
        for ym in sorted(monthly_total.keys()):
            yr   = int(ym[:4])
            tot  = monthly_total[ym]
            nd   = monthly_night[ym]
            nhr  = nd["hours"]
            ntr  = nd["tracks"]
            share = nhr / tot if tot > 0 else 0
            top_artist = (max(nd["artists"], key=nd["artists"].get)
                          if nd["artists"] else "")
            w.writerow([ym, yr, round(nhr, 4), ntr,
                        round(tot, 4), round(share, 4), top_artist])
    print(f"  ✓ {path}  ({len(monthly_total):,} rows)")


# ── OUTPUT 4 — viz_weekday_hourly.csv ───────────────────────────────────────
# One row per (is_weekend, hour).  Aggregated across all time.
# Used for the weekday vs weekend radial clock comparison.
# Columns: bucket, hour, hours_played, track_count, is_night
def write_weekday_hourly(records, out_dir):
    from collections import defaultdict
    agg = defaultdict(lambda: {"hours_played": 0.0, "track_count": 0})

    for r in records:
        bucket = "Weekend" if r["is_weekend"] else "Weekday"
        key = (bucket, r["hour"])
        agg[key]["hours_played"] += r["hours_played"]
        agg[key]["track_count"]  += 1

    path = os.path.join(out_dir, "viz_weekday_hourly.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "hour", "hours_played", "track_count", "is_night"])
        for (bucket, h), v in sorted(agg.items()):
            w.writerow([bucket, h, round(v["hours_played"], 4),
                        v["track_count"], int(h in NIGHT_HOURS)])
    print(f"  ✓ {path}  ({len(agg):,} rows)")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Spotify Hourly CSV Generator ──────────────────────────────")

    # Load raw JSON
    raw = load_all_streams(INPUT_GLOB)

    # Parse all records
    print("Parsing records...")
    parsed = [p for r in raw if (p := parse_record(r)) is not None]
    skipped = len(raw) - len(parsed)
    print(f"  → {len(parsed):,} valid records  ({skipped:,} skipped — under {MIN_MS//1000}s)")

    # Year range sanity check
    years = sorted(set(r["year"] for r in parsed))
    print(f"  → Years covered: {years[0]} – {years[-1]}")

    # Night sessions summary
    night = [r for r in parsed if r["is_night"]]
    night_hrs = sum(r["hours_played"] for r in night)
    print(f"  → Night streams (22h–04h): {len(night):,} plays, "
          f"{night_hrs:.0f} total hours ({100*len(night)/len(parsed):.1f}% of all plays)")

    # Create output dir if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nWriting CSVs to: {OUTPUT_DIR}/")
    write_hourly_intensity(parsed, OUTPUT_DIR)
    write_hour_by_year(parsed, OUTPUT_DIR)
    write_night_sessions(parsed, OUTPUT_DIR)
    write_weekday_hourly(parsed, OUTPUT_DIR)
