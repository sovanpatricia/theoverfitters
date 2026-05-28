"""
genre_balance.py
----------------
Analyses genre distribution in the processed dataset and applies
balancing so no single genre dominates the recommendations.

Balancing strategy:
  - Undersample genres above the target cap
  - Warn about genres below a minimum threshold
  - Save a balanced version ready for embedding

Run after preprocess.py:
    python src/genre_balance.py
"""

import os
import logging
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

RAW_PROCESSED_PATH = os.path.join("data", "processed.csv")
BALANCED_PATH      = os.path.join("data", "balanced.csv")
PLOT_PATH          = os.path.join("data", "genre_distribution.png")

GENRE_COL          = "playlist_genre"     # column name in the dataset
SUBGENRE_COL       = "playlist_subgenre"  # optional, for detailed breakdown

# Balancing config
MIN_SAMPLES_PER_GENRE = 50    # warn if a genre has fewer than this
TARGET_CAP            = 900   # max samples per genre after balancing


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyse_genres(df: pd.DataFrame) -> pd.Series:
    if GENRE_COL not in df.columns:
        raise ValueError(
            f"Column '{GENRE_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    counts = df[GENRE_COL].value_counts()
    total  = len(df)

    print("\n── Genre Distribution (before balancing) ─────────────────")
    print(f"  Total tracks : {total:,}")
    print(f"  Genres found : {len(counts)}\n")
    print(f"  {'Genre':<20} {'Count':>7}  {'%':>6}  Bar")
    print(f"  {'─'*20} {'─'*7}  {'─'*6}  {'─'*30}")
    for genre, count in counts.items():
        pct  = count / total * 100
        bar  = "█" * int(pct / 2)
        flag = "  ⚠ low" if count < MIN_SAMPLES_PER_GENRE else ""
        print(f"  {genre:<20} {count:>7,}  {pct:>5.1f}%  {bar}{flag}")

    print(f"\n  Most common  : {counts.idxmax()} ({counts.max():,})")
    print(f"  Least common : {counts.idxmin()} ({counts.min():,})")
    imbalance = counts.max() / counts.min()
    print(f"  Imbalance ratio (max/min): {imbalance:.1f}x")
    if imbalance > 3:
        print("  ⚠  Imbalance > 3x — balancing recommended")
    else:
        print("  ✓  Distribution looks reasonably balanced")

    if SUBGENRE_COL in df.columns:
        sub_counts = df[SUBGENRE_COL].value_counts()
        print(f"\n── Subgenre breakdown ({len(sub_counts)} subgenres) ────────────")
        for sg, c in sub_counts.items():
            print(f"  {sg:<30} {c:>5,}")

    return counts


# ─── Visualisation ────────────────────────────────────────────────────────────

def plot_distribution(before: pd.Series, after: pd.Series, path: str = PLOT_PATH) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Genre Distribution — Music Mood Matcher", fontsize=14, fontweight="bold")

    colors = ["#7F77DD", "#1D9E75", "#D85A30", "#D4537E", "#378ADD", "#639922",
              "#BA7517", "#E24B4A", "#888780"]

    for ax, series, title in zip(
        axes,
        [before, after],
        ["Before balancing", f"After balancing (cap={TARGET_CAP})"]
    ):
        bars = ax.barh(
            series.index[::-1],
            series.values[::-1],
            color=colors[:len(series)],
            edgecolor="white",
            linewidth=0.5
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Number of tracks")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.spines[["top", "right"]].set_visible(False)

        # Value labels on bars
        for bar, val in zip(bars, series.values[::-1]):
            ax.text(
                bar.get_width() + max(series) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,}",
                va="center", fontsize=9, color="#444"
            )

        ax.set_xlim(0, max(series) * 1.15)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Plot saved to: {path}")


# ─── Balancing ────────────────────────────────────────────────────────────────

def balance(df: pd.DataFrame, cap: int = TARGET_CAP) -> pd.DataFrame:
    """Undersample genres above `cap`. Warn about genres below minimum."""
    if GENRE_COL not in df.columns:
        log.warning(f"No '{GENRE_COL}' column found — skipping balancing.")
        return df

    counts = df[GENRE_COL].value_counts()

    # Warn about underrepresented genres
    for genre, count in counts.items():
        if count < MIN_SAMPLES_PER_GENRE:
            log.warning(
                f"Genre '{genre}' has only {count} samples "
                f"(min recommended: {MIN_SAMPLES_PER_GENRE}). "
                f"Consider merging with a related genre or removing it."
            )

    # Undersample overrepresented genres
    balanced_parts = []
    for genre, group in df.groupby(GENRE_COL):
        if len(group) > cap:
            group = group.sample(n=cap, random_state=42)
            log.info(f"  Undersampled '{genre}': {counts[genre]:,} → {cap}")
        balanced_parts.append(group)

    balanced = pd.concat(balanced_parts).sample(frac=1, random_state=42).reset_index(drop=True)
    return balanced


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load processed data (fall back to raw if processed not available)
    path = RAW_PROCESSED_PATH
    if not os.path.exists(path):
        raw = os.path.join("data", "spotify_lyrics.csv")
        if os.path.exists(raw):
            log.info(f"processed.csv not found — loading raw: {raw}")
            path = raw
        else:
            print(
                f"\n✗ Dataset not found at: {path}\n"
                f"  Run preprocess.py first, or place the raw CSV at data/spotify_lyrics.csv\n"
            )
            return

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} tracks")

    # Analyse before balancing
    counts_before = analyse_genres(df)

    # Balance
    print(f"\n── Balancing (cap={TARGET_CAP} per genre) ────────────────────")
    df_balanced = balance(df)
    counts_after = df_balanced[GENRE_COL].value_counts() if GENRE_COL in df_balanced.columns else counts_before

    print(f"\n── Genre Distribution (after balancing) ──────────────────")
    total = len(df_balanced)
    print(f"  Total tracks : {total:,}")
    for genre, count in counts_after.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {genre:<20} {count:>7,}  {pct:>5.1f}%  {bar}")

    imbalance_after = counts_after.max() / counts_after.min()
    print(f"\n  Imbalance ratio after balancing: {imbalance_after:.1f}x")

    # Plot
    plot_distribution(counts_before, counts_after)

    # Save
    df_balanced.to_csv(BALANCED_PATH, index=False)
    log.info(f"Balanced dataset saved to: {BALANCED_PATH}")
    print(f"\n✓ Done. Balanced dataset: {len(df_balanced):,} tracks → {BALANCED_PATH}")
    print(f"✓ Distribution plot      → {PLOT_PATH}\n")


if __name__ == "__main__":
    main()