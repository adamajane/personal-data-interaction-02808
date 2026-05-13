# A Life in Songs — Data Pipeline

## Overview

This pipeline transforms raw Spotify streaming history (JSON exports) into clean,
aggregated datasets ready for visualization in Observable.

## Pipeline Steps

### Step 1: `01_clean_data.py`

- Loads all JSON files from `data/raw/`
- Filters out podcasts, audiobooks, and non-music content
- Applies minimum play threshold (30 seconds)
- Removes incognito plays
- Parses timestamps and extracts time features
- Profiles the dataset and prints summary stats
- **Outputs:** `data/processed/clean_streams.csv`

### Step 2: `02_enrich_genres.py`

- Reads the clean dataset
- Extracts unique track URIs → fetches artist IDs via Spotify API (batch of 50)
- Fetches artist genres via Spotify API (batch of 50)
- Builds an artist → genres lookup table
- **Outputs:** `data/processed/artist_genres.csv`
- **Requires:** Spotify API credentials in `.env` file

### Step 3: `03_build_viz_data.py`

- Merges clean streams with artist genres
- Builds aggregated datasets for each visualization:
  - `viz_genre_streams.csv` — monthly genre proportions (for streamgraph)
  - `viz_artist_dominance.csv` — top artists per month (for bump chart)
  - `viz_daily_intensity.csv` — daily listening totals (for calendar heatmap)
  - `viz_geo_monthly.csv` — monthly country distribution (for geographic map)
  - `viz_behavior_monthly.csv` — monthly listening behavior patterns (for behavior chart)
- **Outputs:** `data/viz/` directory

### Step 4: `04_generate_hourly.py`

- Loads raw JSON files directly from `data/raw/` (independent of Steps 1–3)
- Applies minimum play threshold (30 seconds) and parses UTC timestamps
- Extracts hour-of-day, weekday, and night-hour (22h–04h) features
- Builds aggregated datasets for hour-based visualizations:
  - `viz_hourly_intensity.csv` — listening per (year_month, hour) (for radial clock and heatmap)
  - `viz_hour_by_year.csv` — listening per (year, hour) (for hour × year chart)
  - `viz_night_sessions.csv` — monthly night-session totals with top night artist
  - `viz_weekday_hourly.csv` — hourly listening split by weekday vs weekend (for radial clock comparison)
- **Outputs:** `data/viz/` directory

## Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy your Spotify JSON files into data/raw/
#    (all Streaming_History_Audio_*.json files)

# 4. Set up Spotify credentials
cp .env.example .env
# Edit .env with your Spotify Client ID and Secret

# 5. Run the pipeline
python 01_clean_data.py
python 02_enrich_genres.py
python 03_build_viz_data.py
python 04_generate_hourly.py
```

## Output

All visualization-ready files end up in `data/viz/` — these are what you load
into your Observable notebook.
