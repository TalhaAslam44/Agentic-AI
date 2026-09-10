"""Dense, lexical and hybrid retrieval over chunks."""

import numpy as np

from chunking import build_chunks
from embedder import Embedder
from lexical import BM25


class HybridIndex:
    """Retrieval over chunks, by BM25, by embedding, or by a fusion of both.

    fusion="rrf"    reciprocal rank fusion, combines ranks
    fusion="linear" min-max normalise both score sets, then blend
    """

    def __init__(self, mode="hybrid", strategy="sentence-450",
                 fusion="rrf", alpha=0.5, rrf_k=60, embedder=None):
        self.mode = mode
        self.strategy = strategy
        self.fusion = fusion
        self.alpha = alpha          # weight on the dense side, linear fusion only
        self.rrf_k = rrf_k
        self.embedder = embedder or Embedder()
        self.chunks = []
        self._bm25 = None
        self._vectors = None

    def build(self, docs):
        self.chunks = build_chunks(docs, self.strategy)
        texts = [c.display for c in self.chunks]

        if self.mode in ("lexical", "hybrid"):
            self._bm25 = BM25().fit(texts)
        if self.mode in ("dense", "hybrid"):
            self._vectors = self.embedder.encode(texts)
        return self

    def _dense_scores(self, query):
        q = self.embedder.encode([query])
        return (self._vectors @ q.T).ravel()

    @staticmethod
    def _minmax(scores):
        """Squash to 0..1 so two incomparable score scales can be added.

        BM25 is unbounded and cosine lives in -1..1, so blending the raw
        numbers would let BM25 drown the dense side entirely.
        """
        lo, hi = scores.min(), scores.max()
        return np.zeros_like(scores) if hi - lo < 1e-12 else (scores - lo) / (hi - lo)

    def _rrf(self, score_lists):
        """Reciprocal rank fusion.

        Each ranker contributes 1/(k + rank) for every item. Only the ordering
        is used, never the magnitude, so no normalisation is needed and one
        ranker with a wild score scale cannot dominate. k dampens the top: a
        larger k makes the ranker's first place less decisive.
        """
        fused = np.zeros(len(self.chunks))
        for scores in score_lists:
            ranks = np.empty(len(scores), dtype=np.int64)
            ranks[np.argsort(-scores)] = np.arange(len(scores))
            fused += 1.0 / (self.rrf_k + ranks + 1)
        return fused

    def _scores(self, query):
        if self.mode == "lexical":
            return self._bm25.scores(query)
        if self.mode == "dense":
            return self._dense_scores(query)

        lex = self._bm25.scores(query)
        dense = self._dense_scores(query)
        if self.fusion == "rrf":
            return self._rrf([lex, dense])
        return self.alpha * self._minmax(dense) + (1 - self.alpha) * self._minmax(lex)

    def search(self, query, k=5, dedupe_docs=False):
        """Return up to k chunks as (chunk, score), best first.

        dedupe_docs keeps only the best chunk per document, which is what you
        want when overlapping chunks would otherwise fill the results with
        near-duplicate text from one source.
        """
        scores = self._scores(query)
        order = np.argsort(-scores)

        results, seen = [], set()
        for i in order:
            chunk = self.chunks[i]
            if dedupe_docs and chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            results.append((chunk, float(scores[i])))
            if len(results) == k:
                break
        return results
