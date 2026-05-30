"""
mood_detector.py
----------------
Hybrid mood detection combining:
  1. Keyword matching  — fast, explicit, handles known terms
  2. Semantic matching — catches synonyms, paraphrases, artist references

The two methods are merged with configurable weights.
Result: a mood profile dict of audio feature ranges.
"""

import re
import logging
from difflib import SequenceMatcher

log = logging.getLogger(__name__)

# ─── Semantic synonym map ─────────────────────────────────────────────────────
# Maps canonical mood keywords → list of semantic synonyms/paraphrases.
# When any synonym appears in the mood text, it maps to the canonical keyword
# and triggers its audio profile.

SEMANTIC_SYNONYMS: dict[str, list[str]] = {
    # Emotional moods
    "sad": [
        "heartbroken", "depressed", "gloomy", "down", "blue", "sorrowful",
        "unhappy", "miserable", "devastated", "broken", "grief", "crying",
        "tearful", "hopeless", "despair", "empty", "lost", "hurt",
    ],
    "happy": [
        "joyful", "cheerful", "upbeat", "elated", "ecstatic", "glad",
        "content", "blissful", "excited", "thrilled", "delighted", "great",
        "wonderful", "fantastic", "amazing", "on top of the world",
    ],
    "melancholic": [
        "bittersweet", "somber", "pensive", "forlorn", "brooding",
        "wistful", "longing", "aching", "heavy hearted", "heavy heart",
        "quiet sadness", "reflective", "contemplative",
    ],
    "nostalgic": [
        "reminiscent", "throwback", "memories", "sentimental", "old days",
        "back in time", "miss the old", "remember when", "used to",
        "childhood", "long ago", "years ago", "old photos",
    ],
    "romantic": [
        "in love", "loving", "tender", "intimate", "affectionate",
        "crushing", "infatuated", "head over heels", "lovestruck",
        "date night", "valentine", "missing someone", "thinking of you",
    ],
    "angry": [
        "frustrated", "furious", "rage", "mad", "irritated", "aggressive",
        "annoyed", "livid", "boiling", "fed up", "pissed", "intense",
        "fired up", "venting",
    ],
    "heartbroken": [
        "dumped", "breakup", "broke up", "ended things", "she left",
        "he left", "they left", "moved on", "cheated", "betrayed",
        "miss you", "can't get over", "still love", "after a breakup",
    ],
    "anxious": [
        "stressed", "nervous", "worried", "overwhelmed", "tense",
        "on edge", "uneasy", "restless", "panicking", "overthinking",
        "can't sleep", "racing thoughts",
    ],
    "euphoric": [
        "on top of the world", "unstoppable", "incredible", "electric",
        "buzzing", "floating", "high on life", "best day", "celebrating",
    ],

    # Energy moods
    "energetic": [
        "pumped", "hyped", "amped", "wired", "motivated", "charged",
        "ready to go", "full of energy", "can't sit still", "buzzing",
        "alive", "awake", "fired up",
    ],
    "chill": [
        "relaxed", "mellow", "laid back", "easy", "peaceful", "serene",
        "tranquil", "unwinding", "winding down", "taking it easy",
        "no stress", "low key", "cozy", "comfortable", "at ease",
    ],

    # Activity/setting moods
    "gym": [
        "workout", "lifting", "training", "exercise", "running",
        "cardio", "crossfit", "gains", "hustle", "grind", "push",
    ],
    "party": [
        "festive", "celebratory", "wild", "turn up", "lit", "clubbing",
        "going out", "night out", "pre-game", "shots", "dance floor",
    ],
    "road trip": [
        "driving", "on the road", "long drive", "highway", "windows down",
        "cruise", "road", "journey", "travelling", "travel",
    ],
    "study": [
        "studying", "focus", "concentration", "work", "working",
        "productive", "deep work", "reading", "writing", "coding",
    ],
    "morning": [
        "waking up", "just woke up", "sunrise", "breakfast", "coffee",
        "start of the day", "early", "fresh start", "new day",
    ],
    "night": [
        "late night", "midnight", "can't sleep", "insomnia", "dark",
        "alone at night", "2am", "3am", "after midnight",
    ],
    "rainy": [
        "rain", "raining", "stormy", "grey day", "cloudy", "overcast",
        "drizzle", "thunderstorm", "dark outside", "wet",
    ],

    # Dance styles
    "dancing": [
        "dance", "want to dance", "feel like dancing", "move my body",
        "hit the floor", "groove", "on the dance floor",
    ],
    "bachata": [
        "bachata", "latin dance", "romantic latin", "slow latin",
    ],
    "salsa": [
        "salsa", "fast latin", "cuban", "puerto rican dance",
    ],

    # Artist/vibe references — maps to closest mood profile
    "sad": [
        "billie eilish vibes", "bon iver vibes", "elliot smith vibes",
        "radiohead mood", "phoebe bridgers vibes", "sufjan stevens",
    ],
    "chill": [
        "lofi vibes", "lo-fi", "ambient", "background music",
        "coffee shop vibes", "cafe music", "indie folk vibes",
        "mac demarco vibes", "tame impala vibes",
    ],
    "energetic": [
        "kanye vibes", "travis scott vibes", "weeknd vibes",
        "pop smoke vibes", "metro boomin vibes",
    ],
    "happy": [
        "pharrell vibes", "bruno mars vibes", "harry styles vibes",
        "doja cat vibes", "lizzo vibes",
    ],
    "romantic": [
        "frank ocean vibes", "daniel caesar vibes", "sza vibes",
        "giveon vibes", "the weeknd romantic",
    ],
}

# ─── Audio profile map (same as in rag.py, kept here for the detector) ────────

AUDIO_PROFILES: dict[str, dict] = {
    "bachata":    {"danceability": (0.60, 1.0), "energy": (0.35, 0.85), "valence": (0.25, 0.85)},
    "salsa":      {"danceability": (0.65, 1.0), "energy": (0.55, 1.00), "valence": (0.45, 1.00)},
    "tango":      {"danceability": (0.50, 0.90),"energy": (0.35, 0.85), "valence": (0.10, 0.65)},
    "dance":      {"danceability": (0.65, 1.0), "energy": (0.55, 1.00), "valence": (0.40, 1.00)},
    "dancing":    {"danceability": (0.65, 1.0), "energy": (0.55, 1.00), "valence": (0.40, 1.00)},
    "energetic":  {"energy": (0.75, 1.00), "danceability": (0.55, 1.00)},
    "hype":       {"energy": (0.80, 1.00), "danceability": (0.65, 1.00), "valence": (0.50, 1.00)},
    "gym":        {"energy": (0.80, 1.00), "danceability": (0.60, 1.00)},
    "workout":    {"energy": (0.80, 1.00), "danceability": (0.60, 1.00)},
    "calm":       {"energy": (0.00, 0.45), "danceability": (0.00, 0.60)},
    "relaxed":    {"energy": (0.00, 0.45), "acousticness": (0.20, 1.00)},
    "chill":      {"energy": (0.00, 0.50), "valence": (0.30, 0.70)},
    "sleep":      {"energy": (0.00, 0.30), "acousticness": (0.40, 1.00)},
    "study":      {"energy": (0.00, 0.50)},
    "sad":        {"valence": (0.00, 0.35), "energy": (0.00, 0.60)},
    "melancholic":{"valence": (0.00, 0.40), "energy": (0.00, 0.55)},
    "nostalgic":  {"valence": (0.20, 0.55), "energy": (0.20, 0.60)},
    "happy":      {"valence": (0.65, 1.00), "energy": (0.50, 1.00)},
    "euphoric":   {"valence": (0.75, 1.00), "energy": (0.70, 1.00), "danceability": (0.60, 1.00)},
    "angry":      {"energy": (0.75, 1.00), "valence": (0.00, 0.40)},
    "anxious":    {"energy": (0.40, 0.80), "valence": (0.00, 0.45)},
    "romantic":   {"valence": (0.40, 0.80), "energy": (0.20, 0.65), "acousticness": (0.10, 0.80)},
    "heartbroken":{"valence": (0.00, 0.35), "energy": (0.15, 0.55)},
    "morning":    {"energy": (0.40, 0.80), "valence": (0.45, 0.85)},
    "night":      {"energy": (0.20, 0.65), "valence": (0.20, 0.65)},
    "rainy":      {"energy": (0.10, 0.55), "valence": (0.10, 0.50)},
    "party":      {"danceability": (0.70, 1.00), "energy": (0.70, 1.00), "valence": (0.55, 1.00)},
    "road trip":  {"energy": (0.55, 0.90), "valence": (0.50, 0.90)},
}


# ─── Fuzzy matching helper ────────────────────────────────────────────────────

def _fuzzy_match(text: str, phrase: str, threshold: float = 0.82) -> bool:
    """Check if phrase appears in text with fuzzy tolerance."""
    if phrase in text:
        return True
    # Slide a window of similar length across the text
    plen = len(phrase)
    for i in range(len(text) - plen + 1):
        window = text[i:i + plen + 3]
        ratio = SequenceMatcher(None, phrase, window).ratio()
        if ratio >= threshold:
            return True
    return False


# ─── Hybrid detector ─────────────────────────────────────────────────────────

def detect_mood_profile(mood: str) -> tuple[dict, list[str]]:
    """
    Hybrid mood detection: keyword + semantic synonym matching.

    Returns:
        (profile, matched_keywords)
        profile: dict of audio feature ranges
        matched_keywords: list of canonical keywords that matched (for logging)
    """
    mood_lower = mood.lower().strip()
    # Normalise punctuation
    mood_clean = re.sub(r"[^\w\s]", " ", mood_lower)

    matched: dict[str, str] = {}  # canonical_keyword → match_type

    # Pass 1 — direct keyword match (fast, highest confidence)
    for keyword in AUDIO_PROFILES:
        if keyword in mood_clean:
            matched[keyword] = "keyword"
            log.info(f"  Keyword match: '{keyword}'")

    # Pass 2 — semantic synonym match
    for canonical, synonyms in SEMANTIC_SYNONYMS.items():
        if canonical in matched:
            continue  # already matched directly
        for syn in synonyms:
            if _fuzzy_match(mood_clean, syn):
                matched[canonical] = f"semantic:'{syn}'"
                log.info(f"  Semantic match: '{syn}' → '{canonical}'")
                break  # one match per canonical is enough

    # Merge profiles of all matched keywords
    merged: dict = {}
    for keyword in matched:
        if keyword not in AUDIO_PROFILES:
            continue
        for feat, (lo, hi) in AUDIO_PROFILES[keyword].items():
            if feat not in merged:
                merged[feat] = (lo, hi)
            else:
                # Intersect ranges when multiple keywords match
                cur_lo, cur_hi = merged[feat]
                new_lo, new_hi = max(cur_lo, lo), min(cur_hi, hi)
                # If intersection is empty, take the union instead
                if new_lo > new_hi:
                    merged[feat] = (min(cur_lo, lo), max(cur_hi, hi))
                else:
                    merged[feat] = (new_lo, new_hi)

    matched_list = [f"{k} ({v})" for k, v in matched.items()]
    return merged, matched_list


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        "I feel like dancing bachata",
        "I'm heartbroken, she left me last night",
        "pumped up and ready to hit the gym",
        "something lofi and chill for studying",
        "woke up feeling blue and gloomy",
        "road trip with windows down, summer vibes",
        "I want something with Billie Eilish vibes",
        "tame impala vibes, something dreamy",
        "can't sleep, 3am overthinking",
        "celebrating my birthday, let's go wild",
        "rainy sunday, looking at old photos",
        "I don't know how I feel",  # no match expected
    ]

    print("── Hybrid Mood Detector Tests ────────────────────────\n")
    for mood in tests:
        profile, keywords = detect_mood_profile(mood)
        print(f"  Input:    \"{mood}\"")
        if keywords:
            print(f"  Matched:  {', '.join(keywords)}")
            print(f"  Profile:  {profile}")
        else:
            print(f"  Matched:  (none — semantic search only)")
        print()