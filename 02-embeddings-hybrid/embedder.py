"""Neural sentence embeddings, with a disk cache.

This is the one place project 2 differs fundamentally from project 1. The
vector no longer comes from counting words; it comes from a transformer trained
so that texts meaning the same thing land in the same direction.

Everything downstream is unchanged, which is the point. Normalise, then dot.
"""

import hashlib
import os
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")


class Embedder:
    """Lazily loads the model, and caches vectors keyed by text hash.

    Embedding is the slow part of building an index. Caching to disk means the
    first run downloads and computes, and every later run is instant, which
    makes it practical to iterate on chunking and fusion without waiting.
    """

    def __init__(self, model_name=MODEL_NAME, cache=True):
        self.model_name = model_name
        self.cache = cache
        self._model = None
        os.makedirs(CACHE_DIR, exist_ok=True)

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"loading {self.model_name} (first run downloads ~90MB)...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self):
        return self.model.get_sentence_embedding_dimension()

    def _cache_path(self, texts):
        h = hashlib.sha256(self.model_name.encode())
        for t in texts:
            h.update(b"\x00")
            h.update(t.encode("utf-8"))
        return os.path.join(CACHE_DIR, f"{h.hexdigest()[:32]}.npy")

    def encode(self, texts, use_cache=None):
        """Return an (n, dim) array of unit-length vectors."""
        if isinstance(texts, str):
            texts = [texts]
        use_cache = self.cache if use_cache is None else use_cache

        path = self._cache_path(texts)
        if use_cache and os.path.exists(path):
            return np.load(path)

        # normalize_embeddings makes every row unit length, so a dot product
        # between any two rows is their cosine similarity.
        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        ).astype(np.float32)

        if use_cache:
            np.save(path, vectors)
        return vectors


if __name__ == "__main__":
    emb = Embedder()
    pairs = [
        ("a dog runs in the park", "a puppy sprints across the grass"),
        ("a dog runs in the park", "the stock market closed lower today"),
        ("how do I fix error 404", "the page could not be found"),
        ("photosynthesis", "plants making sugar from sunlight"),
    ]
    flat = [t for pair in pairs for t in pair]
    vecs = emb.encode(flat)
    print(f"\ndimensions: {vecs.shape[1]}\n")
    for i, (a, b) in enumerate(pairs):
        sim = float(vecs[2 * i] @ vecs[2 * i + 1])
        print(f"  {sim:+.3f}  {a!r}\n          {b!r}")

# embeddings are unit length, so the dot product is cosine similarity. A
# similarity of 1 means the two texts are identical in meaning, 0 means they
# are orthogonal, and -1 means they are diametrically opposed. The model is
# trained so that paraphrases land near 1, and unrelated texts land near 0