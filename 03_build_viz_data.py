"""
03_build_viz_data.py — Build visualization-ready datasets for Observable.

Merges clean streams with artist genres, then produces aggregated files
for each visualization in the notebook.

Inputs:
  - data/processed/clean_streams.csv
  - data/processed/artist_genres.csv

Outputs (all in data/viz/):
  - viz_genre_streams.csv      → Genre streamgraph (monthly proportions)
  - viz_artist_dominance.csv   → Top artists per month (bump chart)
  - viz_daily_intensity.csv    → Daily listening totals (calendar heatmap)
  - viz_geo_monthly.csv        → Monthly country distribution (geographic map)
  - viz_behavior_monthly.csv   → Listening behavior patterns over time
  - viz_discovery_ratio.csv    → New vs. familiar artists per month
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
VIZ_DIR = Path("data/viz")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# How many top genres to include in the streamgraph (rest → "Other")
TOP_N_GENRES = 12

# How many top artists to show per month in the bump chart
TOP_N_ARTISTS_PER_MONTH = 10


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clean streams and artist genres."""
    streams = pd.read_csv(PROCESSED_DIR / "clean_streams.csv", parse_dates=["ts"])
    genres = pd.read_csv(PROCESSED_DIR / "artist_genres.csv")
    print(f"  Streams: {len(streams):,} rows")
    print(f"  Artists with genre data: {len(genres):,}")
    return streams, genres


def merge_genres(streams: pd.DataFrame, genres: pd.DataFrame) -> pd.DataFrame:
    """
    Merge genre data onto streams. Each stream gets a 'primary_genre' —
    the first genre tag for the artist (Spotify lists genres roughly by relevance).

    Also creates an 'all_genres' column with the full pipe-separated list.
    """
    genre_lookup = genres.set_index("artist_name")[["genres"]].to_dict()["genres"]

    # Assign primary genre (first listed) and all genres
    streams["all_genres"] = streams["artist_name"].map(genre_lookup).fillna("")
    streams["primary_genre"] = streams["all_genres"].apply(
        lambda g: g.split("|")[0] if g else "unknown"
    )

    known = (streams["primary_genre"] != "unknown").sum()
    print(f"  Streams with genre info: {known:,}/{len(streams):,} ({known/len(streams)*100:.1f}%)")

    return streams


# ──────────────────────────────────────────────
# Visualization Dataset Builders
# ──────────────────────────────────────────────

def build_genre_streams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly genre proportions for the streamgraph.

    Strategy: Take the top N genres globally by play count,
    bucket everything else as "Other", then compute monthly
    proportions of total ms_played.
    """
    # Find the top genres by total listening time across all months
    genre_time = df.groupby("primary_genre")["ms_played"].sum().sort_values(ascending=False)
    top_genres = genre_time.head(TOP_N_GENRES).index.tolist()

    # Map non-top genres to "Other"
    df = df.copy()
    df["genre_group"] = df["primary_genre"].apply(lambda g: g if g in top_genres else "Other")

    # Monthly totals per genre group
    monthly = (
        df.groupby(["year_month", "genre_group"])["ms_played"]
        .sum()
        .reset_index()
    )

    # Convert to proportions within each month
    month_totals = monthly.groupby("year_month")["ms_played"].transform("sum")
    monthly["proportion"] = monthly["ms_played"] / month_totals

    # Also keep raw hours for tooltip info
    monthly["hours"] = monthly["ms_played"] / 1000 / 3600

    return monthly[["year_month", "genre_group", "ms_played", "proportion", "hours"]]


def build_artist_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Top N artists per month by listening time.
    Returns a long-form table suitable for a bump chart.
    """
    monthly = (
        df.groupby(["year_month", "artist_name"])["ms_played"]
        .sum()
        .reset_index()
    )

    results = []
    for ym, group in monthly.groupby("year_month"):
        top = group.nlargest(TOP_N_ARTISTS_PER_MONTH, "ms_played").copy()
        top["rank"] = range(1, len(top) + 1)
        top["hours"] = top["ms_played"] / 1000 / 3600
        results.append(top)

    return pd.concat(results)[["year_month", "artist_name", "rank", "ms_played", "hours"]]


def build_daily_intensity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily listening totals for the calendar heatmap.
    Each row = one day with total ms, hours, track count, unique artists.
    """
    daily = (
        df.groupby("date")
        .agg(
            ms_played=("ms_played", "sum"),
            track_count=("track_id", "count"),
            unique_artists=("artist_name", "nunique"),
        )
        .reset_index()
    )
    daily["hours"] = daily["ms_played"] / 1000 / 3600
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day_of_week"] = daily["date"].dt.dayofweek  # 0=Mon, 6=Sun
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype(int)

    return daily


def build_geo_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly country distribution for the geographic map.
    Shows where the user was listening from each month.
    """
    geo = (
        df.groupby(["year_month", "country"])
        .agg(
            ms_played=("ms_played", "sum"),
            track_count=("track_id", "count"),
        )
        .reset_index()
    )
    geo["hours"] = geo["ms_played"] / 1000 / 3600
    return geo


def build_behavior_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly listening behavior patterns.
    Categorizes reason_start into active vs. passive engagement.
    """
    # Classify start reasons into engagement types
    active_starts = {"clickrow", "playbtn", "backbtn"}  # user-initiated
    passive_starts = {"trackdone", "fwdbtn", "remote", "appload"}  # system/auto

    df = df.copy()
    df["engagement"] = df["reason_start"].apply(
        lambda r: "active" if r in active_starts else "passive"
    )

    monthly = (
        df.groupby(["year_month", "engagement"])
        .agg(
            stream_count=("track_id", "count"),
            ms_played=("ms_played", "sum"),
        )
        .reset_index()
    )

    # Also add shuffle rate per month
    shuffle_monthly = (
        df.groupby("year_month")["shuffle"]
        .mean()
        .reset_index()
        .rename(columns={"shuffle": "shuffle_rate"})
    )

    # Compute proportion
    month_totals = monthly.groupby("year_month")["stream_count"].transform("sum")
    monthly["proportion"] = monthly["stream_count"] / month_totals
    monthly["hours"] = monthly["ms_played"] / 1000 / 3600

    # Merge shuffle rate
    monthly = monthly.merge(shuffle_monthly, on="year_month", how="left")

    return monthly


def build_discovery_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly ratio of new vs. familiar artists.
    An artist is "new" if they haven't appeared in any previous month.
    Tracks the cumulative artist vocabulary over time.
    """
    df = df.copy()
    df = df.sort_values("ts")

    months = sorted(df["year_month"].unique())
    seen_artists = set()
    rows = []

    for ym in months:
        month_df = df[df["year_month"] == ym]
        month_artists = set(month_df["artist_name"].unique())

        new_artists = month_artists - seen_artists
        familiar_artists = month_artists & seen_artists

        rows.append({
            "year_month": ym,
            "total_artists": len(month_artists),
            "new_artists": len(new_artists),
            "familiar_artists": len(familiar_artists),
            "discovery_ratio": len(new_artists) / len(month_artists) if month_artists else 0,
            "cumulative_artists": len(seen_artists | month_artists),
        })

        seen_artists |= month_artists

    return pd.DataFrame(rows)


def main():
    print("📊 A Life in Songs — Building Visualization Datasets\n")

    # Load
    print("[1/8] Loading data...")
    streams, genres = load_data()

    # Merge
    print("\n[2/8] Merging genre information...")
    df = merge_genres(streams, genres)

    # Build each viz dataset
    print("\n[3/8] Building genre streamgraph data...")
    genre_streams = build_genre_streams(df)
    genre_streams.to_csv(VIZ_DIR / "viz_genre_streams.csv", index=False)
    print(f"  → {len(genre_streams)} rows, {genre_streams['genre_group'].nunique()} genre groups")

    print("\n[4/8] Building artist dominance data...")
    artist_dom = build_artist_dominance(df)
    artist_dom.to_csv(VIZ_DIR / "viz_artist_dominance.csv", index=False)
    print(f"  → {len(artist_dom)} rows")

    print("\n[5/8] Building daily intensity data...")
    daily = build_daily_intensity(df)
    daily.to_csv(VIZ_DIR / "viz_daily_intensity.csv", index=False)
    print(f"  → {len(daily)} days")

    print("\n[6/8] Building geographic data...")
    geo = build_geo_monthly(df)
    geo.to_csv(VIZ_DIR / "viz_geo_monthly.csv", index=False)
    print(f"  → {len(geo)} rows, {geo['country'].nunique()} countries")

    print("\n[7/8] Building behavior patterns data...")
    behavior = build_behavior_monthly(df)
    behavior.to_csv(VIZ_DIR / "viz_behavior_monthly.csv", index=False)
    print(f"  → {len(behavior)} rows")

    print("\n[8/8] Building discovery ratio data...")
    discovery = build_discovery_ratio(df)
    discovery.to_csv(VIZ_DIR / "viz_discovery_ratio.csv", index=False)
    print(f"  → {len(discovery)} months")

    # Summary
    print("\n" + "=" * 60)
    print("✅ All visualization datasets saved to data/viz/")
    print("=" * 60)
    for f in sorted(VIZ_DIR.glob("*.csv")):
        size = f.stat().st_size / 1024
        print(f"  {f.name:<35} {size:>8.1f} KB")
    print("\nReady for Observable! 🚀")


if __name__ == "__main__":
    main()
