"""Search the corpus, showing what each retrieval method returns.

  python search.py "your question"          hybrid results
  python search.py --compare "your question"  all three methods side by side
  python search.py                          interactive
"""

import sys

from corpus import DOCS
from embedder import Embedder
from index import HybridIndex

SNIPPET = 150


def show(title, results):
    print(f"\n  {title}")
    if not results:
        print("    no matching chunks")
        return
    for rank, (chunk, score) in enumerate(results, 1):
        text = chunk.text[:SNIPPET].rsplit(" ", 1)[0]
        print(f"    {rank}. [{score:.4f}] {chunk.title}  ({chunk.chunk_id})")
        print(f"       {text}...")


def main():
    args = [a for a in sys.argv[1:] if a != "--compare"]
    compare = "--compare" in sys.argv
    embedder = Embedder()

    indexes = {}
    if compare:
        for label, kw in [("BM25", dict(mode="lexical")),
                          ("dense", dict(mode="dense")),
                          ("hybrid RRF", dict(mode="hybrid", fusion="rrf"))]:
            indexes[label] = HybridIndex(embedder=embedder, **kw).build(DOCS)
    else:
        indexes["hybrid RRF"] = HybridIndex(
            mode="hybrid", fusion="rrf", embedder=embedder).build(DOCS)

    n_chunks = len(next(iter(indexes.values())).chunks)
    queries = [" ".join(args)] if args else None
    if queries is None:
        print(f"indexed {n_chunks} chunks from {len(DOCS)} documents. blank line to quit.")

    while True:
        if queries is not None:
            if not queries:
                return
            query = queries.pop(0)
        else:
            try:
                query = input("\nquery> ").strip()
            except EOFError:
                return
            if not query:
                return

        for label, index in indexes.items():
            show(label, index.search(query, k=3, dedupe_docs=True))


if __name__ == "__main__":
    main()
