"""A tiny vector index. Same interface a real vector database gives you."""

import numpy as np
from vectorize import TfidfVectorizer, LSA


class VectorIndex:
    """Brute force cosine search over unit-length vectors.

    mode="tfidf"  -> lexical matching only
    mode="lsa"    -> TF-IDF projected into a dense latent space
    mode="hybrid" -> weighted blend of the two scores
    """

    def __init__(self, mode="hybrid", n_components=64, alpha=0.5):
        self.mode = mode
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer()
        self.lsa = LSA(n_components=n_components)
        self.ids = []
        self.titles = []
        self.texts = []
        self._sparse = None   # (N, V) tf-idf
        self._dense = None    # (N, k) lsa

    def build(self, docs):
        """docs: iterable of (id, title, text)."""
        self.ids = [d[0] for d in docs]
        self.titles = [d[1] for d in docs]
        self.texts = [f"{d[1]}. {d[2]}" for d in docs]

        self._sparse = self.vectorizer.fit_transform(self.texts)
        self._dense = self.lsa.fit_transform(self._sparse)
        return self

    def _scores(self, query):
        qs = self.vectorizer.transform([query])
        lex = (self._sparse @ qs.T).ravel()
        if self.mode == "tfidf":
            return lex
        sem = (self._dense @ self.lsa.transform(qs).T).ravel()
        if self.mode == "lsa":
            return sem
        return self.alpha * lex + (1 - self.alpha) * sem

    def search(self, query, k=5, min_score=1e-6):
        """Return up to k matches as (id, title, score), best first.

        Results scoring at or below min_score are dropped. A query sharing no
        vocabulary with the corpus scores zero everywhere, and without this
        guard the tie is broken by storage order, so the caller gets confident
        looking nonsense instead of an honest empty list.
        """
        scores = self._scores(query)
        k = min(k, len(scores))
        # argpartition finds the top k without fully sorting; this is the
        # same trick an ANN index accelerates further.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            (self.ids[i], self.titles[i], float(scores[i]))
            for i in top
            if scores[i] > min_score
        ]
