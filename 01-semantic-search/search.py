"""Interactive search over the corpus.  python search.py "your question here" """

import sys
from corpus import DOCS
from index import VectorIndex


def main():
    index = VectorIndex(mode="hybrid", n_components=64, alpha=0.5).build(DOCS)

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]
    else:
        print("built index over", len(DOCS), "documents. blank line to quit.")
        queries = None

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

        results = index.search(query, k=5)
        if not results:
            print("  no matching documents")
            continue
        for rank, (doc_id, title, score) in enumerate(results, 1):
            print(f"  {rank}. [{score:.3f}] {title}  ({doc_id})")


if __name__ == "__main__":
    main()
