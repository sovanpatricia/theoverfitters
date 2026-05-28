"""
hallucination.py
----------------
Verifies that songs returned by the LLM actually exist in the dataset.
Removes any hallucinated songs before returning results to the user.

Contract:
    verify(recommendations, dataset_df) -> (verified, removed)
"""

import logging
import pandas as pd

log = logging.getLogger(__name__)


class HallucinationGuard:
    """
    Cross-references LLM-generated song suggestions against the dataset index.

    Matching strategy (in order of strictness):
      1. Exact match on both track_name and track_artist
      2. Case-insensitive match
      3. Partial match (title contains or is contained by dataset entry)

    Any song that fails all three checks is flagged as hallucinated and removed.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: the processed/balanced dataset DataFrame.
                Must contain 'track_name' and 'track_artist' columns.
        """
        self._build_index(df)

    def _build_index(self, df: pd.DataFrame) -> None:
        """Build a fast lookup set from the dataset."""
        self._exact: set[tuple[str, str]] = set(
            zip(df["track_name"].str.strip(), df["track_artist"].str.strip())
        )
        self._lower: set[tuple[str, str]] = set(
            zip(df["track_name"].str.strip().str.lower(),
                df["track_artist"].str.strip().str.lower())
        )
        self._titles_lower: set[str] = set(df["track_name"].str.strip().str.lower())
        log.info(f"Hallucination guard index built: {len(self._exact):,} songs")

    def _is_real(self, track_name: str, track_artist: str) -> tuple[bool, str]:
        """
        Returns (is_real: bool, match_type: str).
        """
        name   = str(track_name).strip()
        artist = str(track_artist).strip()

        # Level 1: exact match
        if (name, artist) in self._exact:
            return True, "exact"

        # Level 2: case-insensitive
        if (name.lower(), artist.lower()) in self._lower:
            return True, "case-insensitive"

        # Level 3: partial title match (handles minor truncation by LLM)
        name_lower = name.lower()
        for known_title in self._titles_lower:
            if name_lower in known_title or known_title in name_lower:
                return True, "partial"

        return False, "none"

    def verify(
        self, recommendations: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Verify a list of LLM-generated recommendations.

        Args:
            recommendations: list of dicts with 'track_name' and 'track_artist'

        Returns:
            (verified, removed)
            - verified: songs confirmed to exist in the dataset
            - removed:  hallucinated songs that were stripped out
        """
        verified = []
        removed  = []

        for rec in recommendations:
            name   = rec.get("track_name",   "")
            artist = rec.get("track_artist", "")
            real, match_type = self._is_real(name, artist)

            if real:
                rec["_verified"] = match_type
                verified.append(rec)
                log.debug(f"  ✓ {name!r} by {artist!r} ({match_type})")
            else:
                log.warning(f"  ✗ Hallucinated: {name!r} by {artist!r} — removed")
                removed.append(rec)

        if removed:
            log.warning(
                f"Hallucination guard removed {len(removed)} song(s) "
                f"({len(verified)} verified)"
            )
        else:
            log.info(f"All {len(verified)} recommendations verified ✓")

        return verified, removed


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build a small mock dataset
    mock_df = pd.DataFrame({
        "track_name":   ["Bohemian Rhapsody", "Hotel California", "Stairway to Heaven"],
        "track_artist": ["Queen",             "Eagles",           "Led Zeppelin"],
    })

    guard = HallucinationGuard(mock_df)

    test_recs = [
        {"track_name": "Bohemian Rhapsody",  "track_artist": "Queen",       "reason": "Classic."},
        {"track_name": "bohemian rhapsody",  "track_artist": "queen",       "reason": "Same song, lowercase."},
        {"track_name": "Hotel Californiaaaa","track_artist": "Eagles",      "reason": "Slight typo."},
        {"track_name": "Imaginary Song",     "track_artist": "Fake Artist", "reason": "Does not exist."},
    ]

    verified, removed = guard.verify(test_recs)

    print("\n── Hallucination Guard Test ──────────────────────")
    print(f"  Input      : {len(test_recs)} songs")
    print(f"  Verified   : {len(verified)}")
    print(f"  Removed    : {len(removed)}")
    print("\n  Verified:")
    for s in verified:
        print(f"    ✓ {s['track_name']} — match: {s['_verified']}")
    print("\n  Removed (hallucinated):")
    for s in removed:
        print(f"    ✗ {s['track_name']} by {s['track_artist']}")