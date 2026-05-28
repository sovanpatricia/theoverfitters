"""
rag.py
------
Combines prompt engineering + hybrid retrieval + LM Studio inference
to generate mood-based song recommendations with explanations.

Hybrid retrieval:
  1. Semantic search over lyrics (ChromaDB)
  2. Audio feature filtering based on detected mood keywords

Requires LM Studio running locally:
    - Load Mistral 7B or LLaMA 3.2 3B
    - Start the local server (default: http://localhost:1234)
"""

import os
import sys
import json
import logging
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from embeddings import retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

# ─── LM Studio client ────────────────────────────────────────────────────────

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
LM_MODEL      = os.getenv("LM_MODEL", "local-model")
TOP_K         = int(os.getenv("TOP_K", "5"))

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")


# ─── Mood → Audio feature mapping ────────────────────────────────────────────
#
# Maps mood keywords to Spotify audio feature ranges.
# valence:      0–1  (sad → happy)
# energy:       0–1  (calm → intense)
# danceability: 0–1  (low → high)
#
# We retrieve EXTRA candidates (k * RETRIEVAL_MULTIPLIER) then filter,
# ensuring we always have enough after filtering.

RETRIEVAL_MULTIPLIER = 8   # retrieve 8x more than needed, then filter down

# Keywords that require a specific playlist_genre to be present
GENRE_REQUIREMENTS: dict[str, str] = {
    "bachata": "latin",
    "salsa":   "latin",
    "tango":   "latin",
    "reggaeton":"latin",
    "latin":   "latin",
}

# ─── Dataset coverage (auto-detected) ───────────────────────────────────────
# Dynamically built from the actual dataset at startup.
# Maps user mood keywords → what genre/subgenre/language to look for.

import os as _os
import pandas as _pd

def _build_coverage() -> tuple[set, set, set]:
    """
    Read the dataset and extract all available:
      - genres (e.g. rock, pop, latin)
      - subgenres (e.g. reggaeton, neo soul)
      - languages (e.g. en, es, ro)
    Returns three sets.
    """
    # Find the dataset relative to this file
    base = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    for candidate in [
        _os.path.join(base, "data", "balanced.csv"),
        _os.path.join(base, "data", "processed.csv"),
        _os.path.join(base, "data", "spotify_lyrics.csv"),
    ]:
        if _os.path.exists(candidate):
            try:
                df = _pd.read_csv(candidate, usecols=[
                    c for c in ["playlist_genre","playlist_subgenre","language"]
                    if c in _pd.read_csv(candidate, nrows=0).columns
                ])
                genres    = set(df["playlist_genre"].dropna().str.lower().unique()) if "playlist_genre" in df else set()
                subgenres = set(df["playlist_subgenre"].dropna().str.lower().unique()) if "playlist_subgenre" in df else set()
                languages = set(df["language"].dropna().str.lower().unique()) if "language" in df else set()
                return genres, subgenres, languages
            except Exception:
                pass
    return set(), set(), set()

_GENRES, _SUBGENRES, _LANGUAGES = _build_coverage()

# Keyword → what to look for in the dataset
# Each entry: (search_term, type)  type = genre|subgenre|language
KEYWORD_COVERAGE_MAP: dict[str, tuple[str, str]] = {
    # Genres
    "rock":         ("rock",         "genre"),
    "pop":          ("pop",          "genre"),
    "rap":          ("rap",          "genre"),
    "edm":          ("edm",          "genre"),
    "latin":        ("latin",        "genre"),
    "r&b":          ("r&b",          "genre"),
    "rnb":          ("r&b",          "genre"),
    # Subgenres
    "reggaeton":    ("reggaeton",    "subgenre"),
    "tropical":     ("tropical",     "subgenre"),
    "latin pop":    ("latin pop",    "subgenre"),
    "latin hip hop":("latin hip hop","subgenre"),
    "neo soul":     ("neo soul",     "subgenre"),
    "hard rock":    ("hard rock",    "subgenre"),
    "classic rock": ("classic rock", "subgenre"),
    "trap":         ("trap",         "subgenre"),
    "hip hop":      ("hip hop",      "subgenre"),
    "gangster rap": ("gangster rap", "subgenre"),
    "electropop":   ("electropop",   "subgenre"),
    "house":        ("electro house","subgenre"),
    # Languages
    "spanish":      ("es",           "language"),
    "german":       ("de",           "language"),
    "french":       ("fr",           "language"),
    "portuguese":   ("pt",           "language"),
    "italian":      ("it",           "language"),
    "romanian":     ("ro",           "language"),
    "moldovan":     ("ro",           "language"),
    # Not in dataset
    "bachata":      ("bachata",      "subgenre"),
    "merengue":     ("merengue",     "subgenre"),
    "cumbia":       ("cumbia",       "subgenre"),
    "tango":        ("tango",        "subgenre"),
    "flamenco":     ("flamenco",     "subgenre"),
    "bossa nova":   ("bossa nova",   "subgenre"),
    "jazz":         ("jazz",         "genre"),
    "classical":    ("classical",    "genre"),
    "country":      ("country",      "genre"),
    "folk":         ("folk",         "genre"),
    "blues":        ("blues",        "genre"),
    "metal":        ("metal",        "subgenre"),
    "punk":         ("punk",         "subgenre"),
    "k-pop":        ("k-pop",        "genre"),
    "kpop":         ("k-pop",        "genre"),
    "afrobeats":    ("afrobeats",    "genre"),
    "reggae":       ("reggae",       "genre"),
    "gospel":       ("gospel",       "genre"),
    "drill":        ("drill",        "subgenre"),
    "manele":       ("manele",       "subgenre"),
    "lautareasca":  ("lautareasca",  "subgenre"),
    "doina":        ("doina",        "subgenre"),
}

# Best fallback genre for missing styles
BEST_FALLBACK: dict[str, str] = {
    "bachata":     "latin",
    "merengue":    "latin",
    "cumbia":      "latin",
    "tango":       "latin",
    "flamenco":    "latin",
    "bossa nova":  "latin",
    "jazz":        "r&b",
    "blues":       "r&b",
    "gospel":      "r&b",
    "soul":        "r&b",
    "classical":   "rock",
    "country":     "rock",
    "folk":        "rock",
    "metal":       "rock",
    "hard rock":   "rock",
    "punk":        "rock",
    "indie":       "rock",
    "punk":        "rock",
    "indie":       "rock",
    "k-pop":       "pop",
    "kpop":        "pop",
    "afrobeats":   "latin",
    "reggae":      "latin",
    "drill":       "rap",
    "manele":      "latin",
    "lautareasca": "latin",
    "doina":       "r&b",
    "romanian":    "pop",
    "moldovan":    "pop",
}


def _is_covered(term: str, kind: str) -> bool:
    """Check if a term exists in the dataset for the given type."""
    if kind == "genre":
        return term in _GENRES
    elif kind == "subgenre":
        return term in _SUBGENRES
    elif kind == "language":
        return term in _LANGUAGES
    return False


def get_fallback_notice(mood: str) -> str:
    """
    Dynamically check if the mood mentions something not in the dataset.
    Returns a user-facing notice string, or empty string if all covered.
    """
    mood_lower = mood.lower()
    for keyword, (term, kind) in KEYWORD_COVERAGE_MAP.items():
        if keyword in mood_lower and not _is_covered(term, kind):
            fallback = BEST_FALLBACK.get(keyword, "similar")
            if kind == "language":
                # Special case: language partially available
                if keyword in ("romanian", "moldovan"):
                    return (
                        f"{keyword.title()} music is almost not in our dataset "
                        f"(only Dragostea Din Tei by O-Zone). Showing closest matches instead."
                    )
                return (
                    f"{keyword.title()} language songs are not in our dataset. "
                    f"Showing the closest {fallback} songs instead."
                )
            return (
                f"{keyword.title()} isn't in our dataset. "
                f"Showing the closest {fallback} songs instead."
            )
    return ""


def get_best_fallback_genre(mood: str) -> str:
    """Return the best fallback genre for missing styles in the mood."""
    mood_lower = mood.lower()
    for keyword in KEYWORD_COVERAGE_MAP:
        if keyword in mood_lower:
            term, kind = KEYWORD_COVERAGE_MAP[keyword]
            if not _is_covered(term, kind):
                return BEST_FALLBACK.get(keyword, "")
    return ""


# ─── Pinned songs ─────────────────────────────────────────────────────────────
# Force specific songs into results for certain keywords.

PINNED_SONGS: dict[str, list[dict]] = {
    "romanian": [{
        "track_name":     "Dragostea Din Tei - Original Romanian Version",
        "track_artist":   "O-Zone",
        "lyrics_snippet": "Vrei să pleci dar nu mă, nu mă iei, chipul tău și dragostea din tei",
        "reason":         "The only Romanian song in our dataset — an iconic pop anthem that became a global phenomenon, originally sung in Romanian.",
        "similarity":     1.0,
        "valence":        0.681,
        "energy":         0.966,
        "danceability":   0.813,
        "_pinned":        True,
    }],
    "moldovan": [{
        "track_name":     "Dragostea Din Tei - Original Romanian Version",
        "track_artist":   "O-Zone",
        "lyrics_snippet": "Vrei să pleci dar nu mă, nu mă iei, chipul tău și dragostea din tei",
        "reason":         "O-Zone is actually a Moldovan band — this is the closest match to Moldovan music in our dataset.",
        "similarity":     1.0,
        "valence":        0.681,
        "energy":         0.966,
        "danceability":   0.813,
        "_pinned":        True,
    }],
}


def get_pinned_songs(mood: str) -> list[dict]:
    """Return any songs that must be pinned to the top of results for this mood."""
    mood_lower = mood.lower()
    pinned = []
    seen_keys: set = set()
    for keyword, songs in PINNED_SONGS.items():
        if keyword in mood_lower:
            for s in songs:
                key = (s["track_name"], s["track_artist"])
                if key not in seen_keys:
                    pinned.append(s)
                    seen_keys.add(key)
    return pinned


MOOD_PROFILES: dict[str, dict] = {
    # Dance styles
    "bachata":    {"danceability": (0.60, 1.0), "energy": (0.35, 0.85), "valence": (0.25, 0.85)},
    "salsa":      {"danceability": (0.65, 1.0), "energy": (0.55, 1.00), "valence": (0.45, 1.00)},
    "tango":      {"danceability": (0.50, 0.90),"energy": (0.35, 0.85), "valence": (0.10, 0.65)},
    "dance":      {"danceability": (0.65, 1.0), "energy": (0.55, 1.00), "valence": (0.40, 1.00)},
    "dancing":    {"danceability": (0.65, 1.0), "energy": (0.55, 1.00), "valence": (0.40, 1.00)},
    # Energy moods
    "energetic":  {"energy": (0.75, 1.00), "danceability": (0.55, 1.00)},
    "hype":       {"energy": (0.80, 1.00), "danceability": (0.65, 1.00), "valence": (0.50, 1.00)},
    "gym":        {"energy": (0.80, 1.00), "danceability": (0.60, 1.00)},
    "workout":    {"energy": (0.80, 1.00), "danceability": (0.60, 1.00)},
    "pump":       {"energy": (0.80, 1.00), "danceability": (0.65, 1.00)},
    # Calm moods
    "calm":       {"energy": (0.00, 0.45), "danceability": (0.00, 0.60)},
    "relaxed":    {"energy": (0.00, 0.45), "acousticness": (0.20, 1.00)},
    "chill":      {"energy": (0.00, 0.50), "valence": (0.30, 0.70)},
    "sleep":      {"energy": (0.00, 0.30), "acousticness": (0.40, 1.00)},
    "study":      {"energy": (0.00, 0.50), "instrumentalness": (0.10, 1.00)},
    # Emotional moods
    "sad":        {"valence": (0.00, 0.35), "energy": (0.00, 0.60)},
    "melancholic":{"valence": (0.00, 0.40), "energy": (0.00, 0.55)},
    "nostalgic":  {"valence": (0.20, 0.55), "energy": (0.20, 0.60)},
    "happy":      {"valence": (0.65, 1.00), "energy": (0.50, 1.00)},
    "euphoric":   {"valence": (0.75, 1.00), "energy": (0.70, 1.00), "danceability": (0.60, 1.00)},
    "angry":      {"energy": (0.75, 1.00), "valence": (0.00, 0.40)},
    "romantic":   {"valence": (0.40, 0.80), "energy": (0.20, 0.65), "acousticness": (0.10, 0.80)},
    "heartbroken":{"valence": (0.00, 0.35), "energy": (0.15, 0.55)},
    # Time/setting moods
    "morning":    {"energy": (0.40, 0.80), "valence": (0.45, 0.85)},
    "night":      {"energy": (0.20, 0.65), "valence": (0.20, 0.65)},
    "rainy":      {"energy": (0.10, 0.55), "valence": (0.10, 0.50)},
    "party":      {"danceability": (0.70, 1.00), "energy": (0.70, 1.00), "valence": (0.55, 1.00)},
    "road trip":  {"energy": (0.55, 0.90), "valence": (0.50, 0.90)},
}


def _detect_mood_profile(mood: str) -> dict:
    """
    Scan the mood description for known keywords and merge their
    audio feature ranges. Returns {} if no keywords matched.
    """
    mood_lower = mood.lower()
    merged: dict = {}

    for keyword, profile in MOOD_PROFILES.items():
        if keyword in mood_lower:
            log.info(f"  Mood keyword detected: '{keyword}'")
            for feat, (lo, hi) in profile.items():
                if feat not in merged:
                    merged[feat] = (lo, hi)
                else:
                    # Intersect ranges if multiple keywords match
                    cur_lo, cur_hi = merged[feat]
                    merged[feat] = (max(cur_lo, lo), min(cur_hi, hi))

    return merged


def _apply_audio_filter(
    songs: list[dict], profile: dict, k: int, required_genre: str = ""
) -> list[dict]:
    """
    Filter songs by:
      1. required_genre (playlist_genre must match, e.g. latin for bachata)
      2. audio feature ranges from the detected mood profile
    Falls back gracefully if too few songs pass.
    """
    if not profile and not required_genre:
        return songs[:k]

    def matches(song: dict) -> bool:
        # Genre filter
        if required_genre and song.get("playlist_genre", "") != required_genre:
            return False
        # Audio feature filter
        for feat, (lo, hi) in profile.items():
            val = song.get(feat)
            if val is not None and not (lo <= val <= hi):
                return False
        return True

    filtered = [s for s in songs if matches(s)]

    if len(filtered) >= k:
        log.info(f"Filter: {len(songs)} → {len(filtered)} songs (genre={required_genre or 'any'}, keeping top {k})")
        return filtered[:k]

    # Not enough — try relaxing genre requirement but keep audio filter
    if required_genre and len(filtered) < k:
        log.warning(
            f"Only {len(filtered)}/{k} songs matched genre='{required_genre}'. "
            f"Relaxing genre filter, keeping audio features."
        )
        audio_only = [s for s in songs if all(
            s.get(f) is None or lo <= s[f] <= hi
            for f, (lo, hi) in profile.items()
        )]
        seen = {(s["track_name"], s["track_artist"]) for s in filtered}
        for s in audio_only:
            if len(filtered) >= k: break
            key = (s["track_name"], s["track_artist"])
            if key not in seen:
                filtered.append(s)
                seen.add(key)

    # Still not enough — fill from unfiltered
    if len(filtered) < k:
        log.warning(f"Still only {len(filtered)}/{k} — filling from unfiltered pool.")
        seen = {(s["track_name"], s["track_artist"]) for s in filtered}
        for s in songs:
            if len(filtered) >= k: break
            key = (s["track_name"], s["track_artist"])
            if key not in seen:
                filtered.append(s)
                seen.add(key)

    return filtered[:k]


# ─── Prompt engineering ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert music curator with deep knowledge of lyrics, \
emotions, rhythm, and how music connects to human feelings.

Your task is to recommend songs that match a user's described mood, energy, and dance style.

You will be given:
  1. The user's mood description
  2. A list of candidate songs pre-filtered to match the mood's audio profile,
     with their lyrics snippets, danceability, energy, and valence scores

Important rules:
  - If the user mentions a specific dance style (bachata, salsa, tango, etc.), only recommend
    songs that actually fit that rhythm and style.
  - Pay attention to danceability and energy scores — high danceability means danceable tracks.
  - Only recommend songs from the provided candidate list. Do not invent songs.

IMPORTANT: You MUST recommend ALL candidates provided to you. Do not skip any.
If the list has 5 songs, return exactly 5 songs. If it has 3, return 3. Never return fewer than provided.

Your response MUST be a valid JSON array. Each element must have exactly these fields:
  - "track_name":   string — the song title
  - "track_artist": string — the artist name
  - "reason":       string — 1–2 sentences explaining why this song fits the mood

Return exactly the JSON array, no extra text before or after."""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Mood: I feel melancholic and nostalgic, like looking at old photos.\n\n"
            "Candidates:\n"
            "1. 'The Night Will Always Win' by Manchester Orchestra — "
            "snippet: 'I can't imagine my life without your ghost...' — valence: 0.12, energy: 0.34, danceability: 0.28\n"
            "2. 'Ribs' by Lorde — "
            "snippet: 'You're the only friend I need, sharing beds like little kids...' — valence: 0.20, energy: 0.41, danceability: 0.38"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps([
            {
                "track_name":   "The Night Will Always Win",
                "track_artist": "Manchester Orchestra",
                "reason":       "The haunting imagery of ghosts and absence perfectly mirrors the bittersweet ache of nostalgia, like a melody you can't place but can't forget.",
            },
            {
                "track_name":   "Ribs",
                "track_artist": "Lorde",
                "reason":       "Its dreamy, introspective quality and themes of frozen moments in time resonate deeply with that feeling of longing for a simpler past.",
            },
        ], indent=2),
    },
]


def _build_user_prompt(mood: str, songs: list[dict]) -> str:
    """Format the mood + retrieved songs into a structured prompt."""
    lines = [f"Mood: {mood}\n\nCandidates:"]
    for i, s in enumerate(songs, 1):
        audio = []
        for feat in ["valence", "energy", "danceability"]:
            if feat in s:
                audio.append(f"{feat}: {s[feat]:.2f}")
        audio_str = ", ".join(audio)
        lines.append(
            f"{i}. '{s['track_name']}' by {s['track_artist']} — "
            f"snippet: \"{s['lyrics_snippet'][:200]}\" — {audio_str}"
        )
    return "\n".join(lines)


# ─── Generation ───────────────────────────────────────────────────────────────

def generate_recommendations(mood: str, k: int = TOP_K) -> tuple[list[dict], str]:
    """
    Full hybrid RAG pipeline.
    Returns: (recommendations, fallback_notice)
    fallback_notice is empty string if no fallback needed.
    """
    # Step 1 — Check dataset coverage, pinned songs, detect mood profile
    fallback_notice = get_fallback_notice(mood)
    if fallback_notice:
        log.info(f"Fallback notice: {fallback_notice}")

    pinned = get_pinned_songs(mood)
    if pinned:
        log.info(f"Pinned songs: {[(s['track_name'], s['track_artist']) for s in pinned]}")

    profile = _detect_mood_profile(mood)
    if profile:
        log.info(f"Mood profile detected: {profile}")
    else:
        log.info("No specific mood profile detected — using semantic search only")

    # Step 2 — Detect genre requirement (use fallback if requested genre not in dataset)
    mood_lower = mood.lower()
    required_genre = next(
        (genre for kw, genre in GENRE_REQUIREMENTS.items() if kw in mood_lower), ""
    )
    if not required_genre:
        required_genre = get_best_fallback_genre(mood)
        if required_genre:
            log.info(f"Fallback genre selected: {required_genre}")
    else:
        log.info(f"Genre requirement detected: {required_genre}")

    # Step 3 — Retrieve candidates, filtered by genre directly in ChromaDB
    retrieve_k = k * RETRIEVAL_MULTIPLIER
    log.info(f"Retrieving top-{retrieve_k} candidates (genre={required_genre or 'any'})")

    if required_genre:
        # Retrieve a large pool genre-filtered, then top up with unfiltered if needed
        genre_candidates = retriever.retrieve(query=mood, k=retrieve_k, genre=required_genre)
        log.info(f"Genre-filtered retrieval ({required_genre}): {len(genre_candidates)} songs")

        # Always also get unfiltered pool as backup
        all_candidates = retriever.retrieve(query=mood, k=retrieve_k)

        # Merge: genre-matched first, then fill from unfiltered
        seen = {(s["track_name"], s["track_artist"]) for s in genre_candidates}
        combined = list(genre_candidates)
        for s in all_candidates:
            if (s["track_name"], s["track_artist"]) not in seen:
                combined.append(s)
                seen.add((s["track_name"], s["track_artist"]))
        candidates = combined
    else:
        candidates = retriever.retrieve(query=mood, k=retrieve_k)

    if not candidates:
        log.warning("No candidates retrieved — check ChromaDB index")
        return []

    # Step 4 — Apply audio feature filter, then deduplicate by artist
    candidates = _apply_audio_filter(candidates, profile, k * 3)

    # Deduplicate: max 1 song per artist to avoid repetition
    seen_artists: set = set()
    deduped = []
    for s in candidates:
        artist = s["track_artist"].lower()
        if artist not in seen_artists:
            deduped.append(s)
            seen_artists.add(artist)
        if len(deduped) >= k:
            break

    # If dedup removed too many, fill back from candidates
    if len(deduped) < k:
        seen_keys = {(s["track_name"], s["track_artist"]) for s in deduped}
        for s in candidates:
            if len(deduped) >= k: break
            key = (s["track_name"], s["track_artist"])
            if key not in seen_keys:
                deduped.append(s)
                seen_keys.add(key)

    candidates = deduped
    log.info(f"Final candidates after filtering + dedup: {len(candidates)}")

    # Step 4 — Build prompt
    user_prompt = _build_user_prompt(mood, candidates)

    # Step 5 — Call LM Studio
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user",      "content": user_prompt},
    ]

    log.info("Calling LM Studio…")
    response = client.chat.completions.create(
        model=LM_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=1200,
    )

    raw = response.choices[0].message.content.strip()
    log.info(f"LM Studio raw response: {raw[:300]}")

    recs = _parse_response(raw)
    log.info(f"Parsed {len(recs)} recommendations from LLM")

    # Inject pinned songs at the top, avoiding duplicates
    if pinned:
        pinned_keys = {(s["track_name"], s["track_artist"]) for s in pinned}
        recs = [r for r in recs if (r["track_name"], r["track_artist"]) not in pinned_keys]
        recs = pinned + recs
        recs = recs[:k]  # keep total at k
        log.info(f"After pinning: {len(recs)} recommendations")

    return recs, fallback_notice


def _parse_response(raw: str) -> list[dict]:
    """Parse the LLM JSON response, handling common formatting issues."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        for key in ("recommendations", "songs", "results"):
            if key in data:
                return data[key]
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse LLM response as JSON: {e}")
        log.debug(f"Raw response: {raw[:500]}")

    return []


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__" and "--live" not in sys.argv:
    print("── Mood profile detection test ───────────────────")
    test_moods = [
        "I feel like dancing bachata",
        "I'm sad and melancholic, rainy day",
        "energetic gym session",
        "chill sunday morning coffee",
        "happy party vibes",
    ]
    for m in test_moods:
        profile = _detect_mood_profile(m)
        print(f"  '{m}'")
        print(f"   → {profile if profile else 'no profile (semantic only)'}\n")

    print("✓ Mood detection OK\n")


# ─── Live LM Studio test ──────────────────────────────────────────────────────

def _live_test():
    print("\n── LM Studio live test ───────────────────────────")
    print(f"  Connecting to: {LM_STUDIO_URL}")

    try:
        models = client.models.list()
        print(f"  Models available: {[m.id for m in models.data]}")
    except Exception as e:
        print(f"\n✗ Cannot connect to LM Studio: {e}\n")
        sys.exit(1)

    mock_songs = [
        {"track_name": "Vivir Mi Vida", "track_artist": "Marc Anthony",
         "lyrics_snippet": "Voy a reir, voy a bailar, vivir mi vida la la la",
         "valence": 0.82, "energy": 0.78, "danceability": 0.85},
        {"track_name": "El Perdón", "track_artist": "Nicky Jam",
         "lyrics_snippet": "Baby I need your forgiveness, come dance with me tonight",
         "valence": 0.71, "energy": 0.65, "danceability": 0.80},
        {"track_name": "Danza Kuduro", "track_artist": "Don Omar",
         "lyrics_snippet": "No te canses bailar, sigue el ritmo del tambor",
         "valence": 0.88, "energy": 0.90, "danceability": 0.92},
    ]

    mood = "I feel like dancing bachata"
    profile = _detect_mood_profile(mood)
    print(f"\n  Mood: \"{mood}\"")
    print(f"  Detected profile: {profile}")

    prompt = _build_user_prompt(mood, mock_songs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOT_EXAMPLES,
        {"role": "user",   "content": prompt},
    ]

    print("  Calling LM Studio...\n")
    response = client.chat.completions.create(
        model=LM_MODEL, messages=messages, temperature=0.4, max_tokens=800,
    )

    recs = _parse_response(response.choices[0].message.content.strip())
    if recs:
        print("── Recommendations ───────────────────────────────")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['track_name']} — {r['track_artist']}")
            print(f"     {r.get('reason', '')}\n")
        print(f"✓ {len(recs)} recommendation(s) returned\n")
    else:
        print("✗ No recommendations parsed.")


if __name__ == "__main__" and "--live" in sys.argv:
    _live_test()