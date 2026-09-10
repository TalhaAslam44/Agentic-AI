"""BM25, implemented from scratch on numpy.

Project 1 used TF-IDF. BM25 is what people actually ship, and the difference is
two corrections that both come from the same observation: raw term frequency
lies. It over-rewards repetition, and it over-rewards long documents.
"""

import re
import numpy as np

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "how", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "them", "they", "this", "to", "was", "were",
    "what", "when", "where", "which", "while", "who", "why", "with",
}

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")

SUFFIXES = ("ational", "iveness", "ization", "ations", "ingly", "izing",
            "ition", "ing", "ies", "ied", "ess", "ed", "es", "s")


def stem(word):
    """Crude suffix stripper.

    Not Porter, but enough to connect 'erupting' to 'erupts' and 'address' to
    'addresses' - the exact failures that project 1 ended on. Applied to both
    documents and queries so the vocabularies line up.
    """
    for suffix in SUFFIXES:
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def tokenize(text, use_stemming=True):
    tokens = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    return [stem(t) for t in tokens] if use_stemming else tokens


class BM25:
    """Okapi BM25.

    k1 controls how fast term frequency saturates. At k1=0 a term either
    appears or does not and repetition is ignored entirely; as k1 grows the
    score approaches raw counts. Typical values sit near 1.2 to 1.5.

    b controls length normalisation. At b=0 document length is ignored, at
    b=1 the score is fully divided by relative length. Typical value is 0.75.
    """

    def __init__(self, k1=1.5, b=0.75, use_stemming=True):
        self.k1 = k1
        self.b = b
        self.use_stemming = use_stemming
        self.vocab = {}
        self.idf = None
        self._weights = None   # (N, V) precomputed tf component

    def fit(self, texts):
        docs = [tokenize(t, self.use_stemming) for t in texts]
        n = len(docs)

        df = {}
        for tokens in docs:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        terms = sorted(df)
        self.vocab = {t: i for i, t in enumerate(terms)}

        counts = np.zeros((n, len(terms)), dtype=np.float64)
        for row, tokens in enumerate(docs):
            for term in tokens:
                counts[row, self.vocab[term]] += 1.0

        lengths = counts.sum(axis=1, keepdims=True)
        avgdl = lengths.mean()

        # Probabilistic idf. It can go negative for terms in more than half the
        # collection, so floor it rather than letting common words subtract.
        df_arr = np.array([df[t] for t in terms], dtype=np.float64)
        self.idf = np.maximum(np.log((n - df_arr + 0.5) / (df_arr + 0.5) + 1.0), 1e-6)

        # tf saturation with length normalisation, precomputed once per document.
        denom = counts + self.k1 * (1 - self.b + self.b * lengths / avgdl)
        self._weights = counts * (self.k1 + 1) / np.maximum(denom, 1e-12)
        return self

    def scores(self, query):
        """Raw BM25 score of every document against the query. Unbounded above."""
        cols = [self.vocab[t] for t in tokenize(query, self.use_stemming) if t in self.vocab]
        if not cols:
            return np.zeros(self._weights.shape[0])
        return (self._weights[:, cols] * self.idf[cols]).sum(axis=1)
