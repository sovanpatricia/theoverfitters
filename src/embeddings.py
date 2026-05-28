"""
embeddings.py
-------------
Converts processed song lyrics into vector embeddings and stores
them in ChromaDB for semantic retrieval.

Run once after preprocessing:
    python src/embeddings.py

This builds the vector index at: data/chroma_db/
"""

import os
import logging
import pandas as pd
from tqdm import tqdm
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    import chromadb as ChromaDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)

BALANCED_PATH   = os.path.join("data", "balanced.csv")
PROCESSED_PATH  = os.path.join("data", "processed.csv")
CHROMA_DIR      = os.path.join("data", "chroma_db")
COLLECTION_NAME = "songs"
EMBED_MODEL     = "all-MiniLM-L6-v2"
BATCH_SIZE      = 64
AUDIO_FEATURES  = ["valence", "energy", "danceability", "acousticness", "tempo", "loudness"]


# ─── Lazy imports (heavy libs) ────────────────────────────────────────────────

def _get_embedder():
    from sentence_transformers import SentenceTransformer
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def _get_chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_DIR)


# ─── Index builder ────────────────────────────────────────────────────────────

def build_index(df: pd.DataFrame) -> None:
    """Embed lyrics and store everything in ChromaDB."""

    embedder = _get_embedder()
    client   = _get_chroma_client()

    # Delete existing collection so we can rebuild cleanly
    try:
        client.delete_collection(COLLECTION_NAME)
        log.info("Deleted existing collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    log.info(f"Embedding {len(df):,} songs in batches of {BATCH_SIZE}…")

    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Indexing"):
        batch      = df.iloc[start : start + BATCH_SIZE]
        ids        = [str(i) for i in batch.index.tolist()]
        lyrics     = batch["lyrics"].tolist()
        embeddings = embedder.encode(lyrics, show_progress_bar=False).tolist()

        metadatas = []
        for _, row in batch.iterrows():
            meta: dict = {
                "track_name":   str(row["track_name"]),
                "track_artist": str(row["track_artist"]),
            }
            for feat in AUDIO_FEATURES:
                if feat in row.index:
                    meta[feat] = float(row[feat])
            if "playlist_genre" in row.index:
                meta["playlist_genre"] = str(row["playlist_genre"])
            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=lyrics,
            metadatas=metadatas,
        )

    log.info(f"Index built — {collection.count():,} vectors stored at: {CHROMA_DIR}")


# ─── Retrieval ────────────────────────────────────────────────────────────────

class SongRetriever:
    """
    Wraps ChromaDB + sentence-transformers for semantic song retrieval.

    Contract:
        retrieve(query: str, k: int = 5) -> list[dict]

    Each result dict contains:
        {track_name, track_artist, lyrics_snippet, similarity, ...audio_features}
    """

    def __init__(self) -> None:
        self._embedder: Optional[object]   = None
        self._collection: Optional[object] = None

    def _load(self) -> None:
        if self._embedder is None:
            self._embedder = _get_embedder()
        if self._collection is None:
            client = _get_chroma_client()
            self._collection = client.get_collection(COLLECTION_NAME)
            log.info(f"Loaded collection: {self._collection.count():,} songs")  # type: ignore[union-attr]

    def retrieve(self, query: str, k: int = 5, genre: str = "") -> list[dict]:
        """
        Embed the mood query and return top-k most similar songs.

        Args:
            query: natural language mood description
            k:     number of songs to return
            genre: if set, only return songs from this playlist_genre

        Returns:
            List of song dicts sorted by relevance (closest first)
        """
        self._load()

        query_embedding = self._embedder.encode([query]).tolist()  # type: ignore[union-attr]

        # Build ChromaDB where clause for genre filtering
        where = {"playlist_genre": genre} if genre else None

        query_kwargs: dict = dict(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)  # type: ignore[union-attr]

        ids_list  = results["ids"][0]
        metas     = results["metadatas"][0]
        docs      = results["documents"][0]
        dists     = results["distances"][0]

        songs = []
        for i in range(len(ids_list)):
            meta    = metas[i]
            doc     = docs[i]
            dist    = dists[i]

            snippet = doc[:300].strip()
            if len(doc) > 300:
                snippet += "…"

            song: dict = {
                "track_name":     meta["track_name"],
                "track_artist":   meta["track_artist"],
                "lyrics_snippet": snippet,
                "similarity":     round(1 - dist, 4),
            }
            # Add audio features and genre if stored
            for key in AUDIO_FEATURES + ["playlist_genre"]:
                if key in meta:
                    song[key] = meta[key]

            songs.append(song)

        return songs

    def filter_by_audio(
        self,
        songs: list[dict],
        valence_min: float = 0.0,
        valence_max: float = 1.0,
        energy_min:  float = 0.0,
        energy_max:  float = 1.0,
    ) -> list[dict]:
        """
        Optional post-retrieval filter using Spotify audio features.
        Useful for mood hints like 'energetic' or 'calm'.
        """
        return [
            s for s in songs
            if valence_min <= s.get("valence", 0.5) <= valence_max
            and energy_min  <= s.get("energy",  0.5) <= energy_max
        ]


# ─── Singleton retriever ──────────────────────────────────────────────────────

retriever = SongRetriever()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = BALANCED_PATH if os.path.exists(BALANCED_PATH) else PROCESSED_PATH
    if not os.path.exists(path):
        print(f"\n✗ Dataset not found at {path}. Run preprocess.py first.\n")
        raise SystemExit(1)

    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} tracks from {path}")

    build_index(df)
    print(f"\n✓ Index built successfully at: {CHROMA_DIR}\n")