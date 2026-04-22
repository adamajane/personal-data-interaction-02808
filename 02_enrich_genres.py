"""
02_enrich_genres.py — Fetch artist genres from Spotify API.

Strategy:
  1. Read clean_streams.csv
  2. Pick one track per unique artist (minimizes API calls)
  3. Batch-fetch tracks (50 at a time) to get Spotify artist IDs
  4. Batch-fetch artists (50 at a time) to get genre tags
  5. Output artist → genres mapping as CSV

Requires: .env file with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
Output: data/processed/artist_genres.csv

Rate limits: Spotify allows ~30 req/sec with a proper token.
We add small delays to be safe.
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
CLEAN_FILE = PROCESSED_DIR / "clean_streams.csv"
OUT_FILE = PROCESSED_DIR / "artist_genres.csv"
BATCH_SIZE = 50  # Spotify's max per request
DELAY_BETWEEN_BATCHES = 0.1  # seconds, conservative


# ──────────────────────────────────────────────
# Spotify Auth — Client Credentials Flow
# ──────────────────────────────────────────────
def get_spotify_token() -> str:
    """Get an access token using client credentials flow."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise ValueError(
            "Missing Spotify credentials!\n"
            "Make sure .env file exists with SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET"
        )

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print(f"  ✅ Got Spotify access token")
    return token


# ──────────────────────────────────────────────
# Batch API Helpers
# ──────────────────────────────────────────────
def batch_get_tracks(track_ids: list[str], token: str) -> dict:
    """
    Fetch tracks in batches of 50 to extract artist IDs.
    Returns: {artist_name: artist_id} mapping
    """
    headers = {"Authorization": f"Bearer {token}"}
    artist_map = {}  # artist_name -> artist_id

    batches = [track_ids[i:i + BATCH_SIZE] for i in range(0, len(track_ids), BATCH_SIZE)]

    print(f"  Fetching {len(track_ids)} tracks in {len(batches)} batches...")
    for batch in tqdm(batches, desc="  Tracks"):
        ids_str = ",".join(batch)
        resp = requests.get(
            f"https://api.spotify.com/v1/tracks?ids={ids_str}",
            headers=headers,
        )

        if resp.status_code == 429:
            # Rate limited — wait and retry
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"\n  ⚠️ Rate limited, waiting {wait}s...")
            time.sleep(wait)
            resp = requests.get(
                f"https://api.spotify.com/v1/tracks?ids={ids_str}",
                headers=headers,
            )

        if resp.status_code != 200:
            print(f"\n  ⚠️ Track batch failed (status {resp.status_code}), skipping...")
            continue

        tracks = resp.json().get("tracks", [])
        for track in tracks:
            if track and track.get("artists"):
                # Use the primary (first) artist
                primary = track["artists"][0]
                artist_map[primary["name"]] = primary["id"]

        time.sleep(DELAY_BETWEEN_BATCHES)

    return artist_map


def batch_get_artist_genres(artist_ids: list[str], token: str) -> dict:
    """
    Fetch artist genres in batches of 50.
    Returns: {artist_id: [genre1, genre2, ...]}
    """
    headers = {"Authorization": f"Bearer {token}"}
    genre_map = {}  # artist_id -> [genres]

    batches = [artist_ids[i:i + BATCH_SIZE] for i in range(0, len(artist_ids), BATCH_SIZE)]

    print(f"  Fetching genres for {len(artist_ids)} artists in {len(batches)} batches...")
    for batch in tqdm(batches, desc="  Artists"):
        ids_str = ",".join(batch)
        resp = requests.get(
            f"https://api.spotify.com/v1/artists?ids={ids_str}",
            headers=headers,
        )

        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 5))
            print(f"\n  ⚠️ Rate limited, waiting {wait}s...")
            time.sleep(wait)
            resp = requests.get(
                f"https://api.spotify.com/v1/artists?ids={ids_str}",
                headers=headers,
            )

        if resp.status_code != 200:
            print(f"\n  ⚠️ Artist batch failed (status {resp.status_code}), skipping...")
            continue

        artists = resp.json().get("artists", [])
        for artist in artists:
            if artist:
                genre_map[artist["id"]] = artist.get("genres", [])

        time.sleep(DELAY_BETWEEN_BATCHES)

    return genre_map


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────
def main():
    print("🎸 A Life in Songs — Genre Enrichment\n")

    # Load clean data
    print("[1/4] Loading clean streams...")
    df = pd.read_csv(CLEAN_FILE)
    print(f"  {len(df):,} streams, {df['artist_name'].nunique():,} unique artists")

    # Get one representative track per artist (the most-played one)
    # This minimizes API calls — we only need one track to get the artist ID.
    print("\n[2/4] Selecting representative tracks per artist...")
    artist_tracks = (
        df.groupby("artist_name")["track_id"]
        .agg(lambda x: x.value_counts().index[0])  # most common track per artist
        .reset_index()
        .rename(columns={"track_id": "rep_track_id"})
    )
    print(f"  {len(artist_tracks)} artists to look up")

    # Authenticate
    print("\n[3/4] Authenticating with Spotify API...")
    token = get_spotify_token()

    # Step A: track IDs → artist IDs
    print("\n[4/4] Fetching data from Spotify API...")
    print("\n  --- Phase A: Tracks → Artist IDs ---")
    artist_id_map = batch_get_tracks(
        artist_tracks["rep_track_id"].tolist(), token
    )
    print(f"  Got artist IDs for {len(artist_id_map)} artists")

    # Step B: artist IDs → genres
    print("\n  --- Phase B: Artist IDs → Genres ---")
    unique_artist_ids = list(set(artist_id_map.values()))
    genre_map = batch_get_artist_genres(unique_artist_ids, token)
    print(f"  Got genres for {len(genre_map)} artists")

    # Build final mapping: artist_name → genres
    rows = []
    for _, row in artist_tracks.iterrows():
        name = row["artist_name"]
        artist_id = artist_id_map.get(name)
        genres = genre_map.get(artist_id, []) if artist_id else []
        rows.append({
            "artist_name": name,
            "artist_id": artist_id or "",
            "genres": "|".join(genres) if genres else "",  # pipe-separated
            "genre_count": len(genres),
        })

    result = pd.DataFrame(rows)

    # Stats
    has_genres = result[result["genre_count"] > 0]
    no_genres = result[result["genre_count"] == 0]
    print(f"\n📊 Genre Enrichment Results:")
    print(f"  Artists with genres:    {len(has_genres):,} ({len(has_genres)/len(result)*100:.1f}%)")
    print(f"  Artists without genres: {len(no_genres):,} ({len(no_genres)/len(result)*100:.1f}%)")

    # Show top genres
    all_genres = []
    for g in has_genres["genres"]:
        all_genres.extend(g.split("|"))
    from collections import Counter
    top_genres = Counter(all_genres).most_common(20)
    print(f"\n  Top 20 genres across all artists:")
    for genre, count in top_genres:
        print(f"    {genre}: {count}")

    # Save
    result.to_csv(OUT_FILE, index=False)
    print(f"\n✅ Saved artist genres to {OUT_FILE}")
    print(f"   {len(result)} artists")


if __name__ == "__main__":
    main()
