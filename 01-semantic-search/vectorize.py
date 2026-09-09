"""TF-IDF and LSA embeddings, built from scratch on numpy.

The point of doing this by hand: an "embedding" is nothing mystical. It is a
vector whose direction encodes meaning. Here the direction comes from word
statistics; in project 2 it will come from a neural net. Everything downstream
(similarity, ranking, evaluation) is identical either way.
"""

import re
import numpy as np

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has",
    "have", "how", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "them", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "while", "who", "why", "with",
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")


def tokenize(text):
    """Lowercase, split on word characters, drop stopwords and 1-char tokens."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class TfidfVectorizer:
    """Bag of words with inverse document frequency weighting."""

    def __init__(self, min_df=1):
        self.min_df = min_df
        self.vocab = {}       # term -> column index
        self.idf = None       # (V,) float array

    def fit(self, docs):
        df = {}
        for doc in docs:
            for term in set(tokenize(doc)):
                df[term] = df.get(term, 0) + 1

        terms = sorted(t for t, c in df.items() if c >= self.min_df)
        self.vocab = {t: i for i, t in enumerate(terms)}

        n = len(docs)
        counts = np.array([df[t] for t in terms], dtype=np.float64)
        # Smoothed idf: +1 inside the log keeps weights positive, +1 on the
        # counts avoids dividing by zero for terms seen in every document.
        self.idf = np.log((1.0 + n) / (1.0 + counts)) + 1.0
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)), dtype=np.float64)
        for row, doc in enumerate(docs):
            for term in tokenize(doc):
                col = self.vocab.get(term)
                if col is not None:
                    X[row, col] += 1.0
        # Sublinear term frequency: the 10th mention of a word says much less
        # than the 2nd, so damp raw counts before applying idf.
        np.log1p(X, out=X)
        X *= self.idf
        return l2_normalize(X)

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)


def l2_normalize(X, eps=1e-12):
    """Scale each row to unit length so a dot product equals cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


class LSA:
    """Latent semantic analysis: truncated SVD over the TF-IDF matrix.

    TF-IDF only matches literal words, so "car" and "automobile" score zero
    against each other. Projecting onto the top singular directions mixes
    co-occurring terms together, which is the crudest possible form of the
    semantic generalisation that neural embeddings do properly.
    """

    def __init__(self, n_components=64):
        self.n_components = n_components
        self.components = None   # (V, k) projection matrix

    def fit(self, X):
        k = min(self.n_components, min(X.shape) - 1)
        # Vt rows are the right singular vectors: directions in term space.
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        self.components = Vt[:k].T
        return self

    def transform(self, X):
        return l2_normalize(X @ self.components)

    def fit_transform(self, X):
        return self.fit(X).transform(X)
