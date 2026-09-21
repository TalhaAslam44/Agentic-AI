"""Bridge to the Project 2 retriever, so we do not rebuild it here.

Project 2 already solved search. This file just imports it and wraps it in one
simple function. Reusing working code instead of copying it is the habit.
"""

import os
import sys

# Project 2 lives in a sibling folder whose name has dashes and digits, so
# Python cannot import it as a normal package. Adding the folder to sys.path
# lets us import the modules inside it directly.
_PROJECT_2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "02-embeddings-hybrid")
if _PROJECT_2 not in sys.path:
    sys.path.insert(0, _PROJECT_2)

from corpus import DOCS            # noqa: E402  (import after sys.path edit)
from index import HybridIndex      # noqa: E402

# Built once when this module is first imported. Embedding every chunk takes a
# few seconds; the cache in Project 2 makes later runs fast.
_index = None


def get_index():
    global _index
    if _index is None:
        print("building index over Project 2 corpus...")
        _index = HybridIndex(
            mode="hybrid",
            fusion="linear",
            alpha=0.7,          # the best setting found in Project 2
            strategy="fixed-400",
        ).build(DOCS)
        print(f"indexed {len(_index.chunks)} chunks from {len(DOCS)} documents\n")
    return _index


def search(query, k=3):
    """Return the k best chunks for a query, as a list of (chunk, score)."""
    return get_index().search(query, k=k, dedupe_docs=True)


if __name__ == "__main__":
    for chunk, score in search("why is attention quadratic", k=3):
        print(f"[{score:.3f}] {chunk.chunk_id}")
        print(f"  {chunk.text[:120]}...\n")


# references:
# - Project 2: 02-embeddings-hybrid/corpus.py, index.py