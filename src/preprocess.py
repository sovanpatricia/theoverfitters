"""
preprocess.py
-------------
Loads the Spotify Audio Features + Lyrics dataset from Kaggle,
cleans it, and saves a processed version ready for embedding.

Dataset: https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs
Download the CSV and place it at: data/spotify_lyrics.csv
"""

import os
import re
import logging
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

RAW_PATH       = os.path.join("data", "spotify_lyrics.csv")
PROCESSED_PATH = os.path.join("data", "processed.csv")

# Spotify audio features we keep for structured mood filtering
AUDIO_FEATURES = [
    "valence",       # 0–1: musical positiveness (sad → happy)
    "energy",        # 0–1: intensity and activity
    "danceability",  # 0–1: rhythmic suitability
    "acousticness",  # 0–1: acoustic vs electronic
    "tempo",         # BPM
    "loudness",      # dB (will be normalised)
]

# Minimum lyric length (characters) to be useful for RAG
MIN_LYRIC_LENGTH = 100

# Maximum lyric length to keep (chars) — avoids oversized chunks
MAX_LYRIC_LENGTH = 3000

# Explicit content column — drop if flagged
EXPLICIT_COL = "explicit"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_lyrics(text: str) -> str:
    """Remove section tags like [Verse 1], [Chorus], extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[.*?\]", "", text)          # remove [Verse 1], [Chorus] etc.
    text = re.sub(r"\n{3,}", "\n\n", text)        # collapse excessive newlines
    text = re.sub(r"[ \t]+", " ", text)           # collapse spaces/tabs
    return text.strip()


def is_english(text: str) -> bool:
    """Best-effort English detection without heavy dependencies."""
    if not isinstance(text, str) or len(text) < 20:
        return False
    # Simple heuristic: ratio of ASCII letters vs total chars
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    total_letters = sum(1 for c in text if c.isalpha())
    if total_letters == 0:
        return False
    return (ascii_letters / total_letters) > 0.85


def normalise_audio_features(df: pd.DataFrame) -> pd.DataFrame:
    """MinMax-scale tempo and loudness to [0, 1]; others are already in range."""
    scaler = MinMaxScaler()
    cols_to_scale = [c for c in ["tempo", "loudness"] if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        log.info(f"Normalised columns: {cols_to_scale}")
    return df


# ─── Main pipeline ────────────────────────────────────────────────────────────

def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    log.info(f"Loading raw dataset from: {path}")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    log.info(f"Columns: {list(df.columns)}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    original_count = len(df)

    # ── 1. Rename columns to consistent snake_case ──────────────────────────
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # ── 2. Drop rows with missing lyrics or track name ──────────────────────
    required = ["lyrics", "track_name", "track_artist"]
    missing_before = len(df)
    df = df.dropna(subset=required)
    log.info(f"Dropped {missing_before - len(df):,} rows with missing lyrics/name/artist")

    # ── 3. Clean lyrics ──────────────────────────────────────────────────────
    df["lyrics"] = df["lyrics"].apply(clean_lyrics)

    # ── 4. Filter by lyric length ────────────────────────────────────────────
    before = len(df)
    df = df[df["lyrics"].str.len().between(MIN_LYRIC_LENGTH, MAX_LYRIC_LENGTH)]
    log.info(f"Dropped {before - len(df):,} rows outside lyric length range "
             f"({MIN_LYRIC_LENGTH}–{MAX_LYRIC_LENGTH} chars)")

    # ── 5. Filter to English lyrics ─────────────────────────────────────────
    before = len(df)
    if "language" in df.columns:
        df = df[df["language"] == "en"]
        log.info(f"Dropped {before - len(df):,} non-English rows (using language column)")
    else:
        df = df[df["lyrics"].apply(is_english)]
        log.info(f"Dropped {before - len(df):,} non-English rows (using heuristic)")

    # ── 6. Drop explicit tracks ──────────────────────────────────────────────
    if EXPLICIT_COL in df.columns:
        before = len(df)
        df = df[df[EXPLICIT_COL] == False]
        log.info(f"Dropped {before - len(df):,} explicit tracks")

    # ── 7. Keep only relevant columns ───────────────────────────────────────
    keep_cols = ["track_name", "track_artist", "lyrics",
                 "playlist_genre", "playlist_subgenre"] + \
                [f for f in AUDIO_FEATURES if f in df.columns]
    missing_features = [f for f in AUDIO_FEATURES if f not in df.columns]
    if missing_features:
        log.warning(f"Audio features not found in dataset: {missing_features}")
    df = df[[c for c in keep_cols if c in df.columns]]

    # ── 8. Drop duplicates ───────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["track_name", "track_artist"])
    log.info(f"Dropped {before - len(df):,} duplicate tracks")

    # ── 9. Normalise audio features ─────────────────────────────────────────
    df = normalise_audio_features(df)

    # ── 10. Reset index ──────────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    log.info(f"Final dataset: {len(df):,} tracks (from {original_count:,} original)")
    return df


def save(df: pd.DataFrame, path: str = PROCESSED_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved processed dataset to: {path}")


def describe(df: pd.DataFrame) -> None:
    """Print a quick summary of the processed dataset."""
    print("\n── Dataset Summary ──────────────────────────────")
    print(f"  Tracks     : {len(df):,}")
    print(f"  Columns    : {list(df.columns)}")
    print(f"\n── Audio Feature Ranges (after normalisation) ───")
    audio_cols = [c for c in AUDIO_FEATURES if c in df.columns]
    print(df[audio_cols].describe().round(3).to_string())
    print(f"\n── Sample Tracks ────────────────────────────────")
    print(df[["track_name", "track_artist"]].sample(min(5, len(df))).to_string(index=False))
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_raw = load_raw()
    df_clean = preprocess(df_raw)
    describe(df_clean)
    save(df_clean)
    print("✓ Preprocessing complete. Output saved to:", PROCESSED_PATH)