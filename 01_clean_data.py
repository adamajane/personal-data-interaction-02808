"""
01_clean_data.py — Load, clean, and profile Spotify streaming history.

Reads all Streaming_History_Audio_*.json files from data/raw/,
filters to music-only plays above a minimum duration threshold,
and outputs a clean CSV for downstream processing.

Output: data/processed/clean_streams.csv
"""

import json
import glob
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration — tweak these if needed
# ──────────────────────────────────────────────
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum play duration in milliseconds to count as an intentional listen.
# 30 seconds (30000ms) is a common threshold — it's also what Spotify uses
# internally to count a "stream" for royalty purposes.
MIN_MS_PLAYED = 30_000
MAX_MS_PLAYED = 3_600_000  # 60 minutes — anything above is a stuck session


# Whether to exclude incognito plays
EXCLUDE_INCOGNITO = True


def load_all_json(raw_dir: Path) -> list[dict]:
    """Load and merge all Streaming_History_Audio_*.json files."""
    all_records = []
    files = sorted(raw_dir.glob("Streaming_History_Audio_*.json"))

    if not files:
        raise FileNotFoundError(
            f"No Streaming_History_Audio_*.json files found in {raw_dir}/\n"
            "Make sure you've copied your Spotify export files into data/raw/"
        )

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            records = json.load(fh)
            all_records.extend(records)
            print(f"  Loaded {f.name}: {len(records):,} records")

    print(f"  Total raw records: {len(all_records):,}")
    return all_records


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to music-only, apply duration threshold, parse timestamps."""

    initial = len(df)

    # --- Filter to music only ---
    # Music rows have a track name but no episode or audiobook fields.
    # Some rows have all nulls (failed plays) — drop those too.
    is_music = (
        df["master_metadata_track_name"].notna()
        & df["episode_name"].isna()
        & df["audiobook_title"].isna()
    )
    df = df[is_music].copy()
    print(
        f"  After music-only filter: {len(df):,} (dropped {initial - len(df):,} non-music rows)"
    )

    # --- Minimum play duration ---
    before = len(df)
    df = df[df["ms_played"] >= MIN_MS_PLAYED].copy()
    print(
        f"  After min duration ({MIN_MS_PLAYED/1000:.0f}s): {len(df):,} (dropped {before - len(df):,} short plays)"
    )

    # --- Maximum play duration (catches stuck sessions) ---
    before = len(df)
    df = df[df["ms_played"] <= MAX_MS_PLAYED].copy()
    print(
        f"  After max duration ({MAX_MS_PLAYED/1000/60:.0f}min): {len(df):,} (dropped {before - len(df):,} stuck sessions)"
    )

    # --- Incognito filter ---
    if EXCLUDE_INCOGNITO:
        before = len(df)
        df = df[~df["incognito_mode"]].copy()
        print(
            f"  After incognito filter: {len(df):,} (dropped {before - len(df):,} incognito plays)"
        )

    # --- Parse timestamps and extract time features ---
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["year_month"] = df["ts"].dt.strftime("%Y-%m")  # "2021-03"
    df["week"] = df["ts"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["ts"].dt.day_name()
    df["hour"] = df["ts"].dt.hour

    # --- Extract track ID from URI (useful for API calls later) ---
    df["track_id"] = df["spotify_track_uri"].str.replace(
        "spotify:track:", "", regex=False
    )

    # --- Rename long column names for convenience ---
    df = df.rename(
        columns={
            "master_metadata_track_name": "track_name",
            "master_metadata_album_artist_name": "artist_name",
            "master_metadata_album_album_name": "album_name",
            "conn_country": "country",
        }
    )

    # --- Select columns we actually need ---
    keep_cols = [
        "ts",
        "date",
        "year",
        "month",
        "year_month",
        "week",
        "day_of_week",
        "hour",
        "track_name",
        "artist_name",
        "album_name",
        "track_id",
        "spotify_track_uri",
        "country",
        "platform",
        "ms_played",
        "reason_start",
        "reason_end",
        "shuffle",
        "skipped",
        "offline",
    ]
    df = df[keep_cols].copy()

    # --- Sort by timestamp ---
    df = df.sort_values("ts").reset_index(drop=True)

    return df


def profile(df: pd.DataFrame):
    """Print a summary profile of the cleaned dataset."""
    print("\n" + "=" * 60)
    print("📊 DATASET PROFILE")
    print("=" * 60)

    total_hours = df["ms_played"].sum() / 1000 / 3600
    print(f"  Total streams:      {len(df):,}")
    print(f"  Date range:         {df['date'].min()} → {df['date'].max()}")
    print(
        f"  Total listening:    {total_hours:,.0f} hours ({total_hours/24:,.0f} days)"
    )
    print(f"  Unique tracks:      {df['track_id'].nunique():,}")
    print(f"  Unique artists:     {df['artist_name'].nunique():,}")
    print(f"  Unique albums:      {df['album_name'].nunique():,}")
    print(
        f"  Countries:          {df['country'].nunique()} — {df['country'].value_counts().head(5).to_dict()}"
    )

    print(f"\n  ms_played stats:")
    ms = df["ms_played"]
    print(f"    Min:    {ms.min():,} ms ({ms.min()/1000:.0f}s)")
    print(f"    Median: {ms.median():,.0f} ms ({ms.median()/1000:.0f}s)")
    print(f"    Mean:   {ms.mean():,.0f} ms ({ms.mean()/1000:.0f}s)")
    print(f"    Max:    {ms.max():,} ms ({ms.max()/1000/60:.0f} min)")

    print(f"\n  Plays per year:")
    yearly = df.groupby("year").size()
    for year, count in yearly.items():
        print(f"    {year}: {count:,}")

    print(f"\n  Top 10 artists (by play count):")
    top_artists = df["artist_name"].value_counts().head(10)
    for artist, count in top_artists.items():
        print(f"    {artist}: {count:,}")

    print(f"\n  Start reasons: {df['reason_start'].value_counts().to_dict()}")
    print(f"  End reasons:   {df['reason_end'].value_counts().to_dict()}")
    print("=" * 60)


def main():
    print("🎵 A Life in Songs — Data Cleaning Pipeline\n")

    # Step 1: Load
    print("[1/3] Loading raw JSON files...")
    records = load_all_json(RAW_DIR)

    # Step 2: Clean
    print("\n[2/3] Cleaning and filtering...")
    df = pd.DataFrame(records)
    df = clean(df)

    # Step 3: Profile & Export
    profile(df)

    out_path = OUT_DIR / "clean_streams.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved clean dataset to {out_path}")
    print(f"   {len(df):,} rows × {len(df.columns)} columns")


if __name__ == "__main__":
    main()
