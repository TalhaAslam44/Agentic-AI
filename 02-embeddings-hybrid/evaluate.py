"""Compare retrieval methods, broken out by query type.

The headline number is the least interesting output here. What matters is the
gap between the semantic and lexical columns, which is the entire argument for
hybrid retrieval.
"""

import sys
import time

from corpus import DOCS
from embedder import Embedder
from index import HybridIndex
from queries import GROUPS, ALL


def evaluate(index, queries, k=5):
    """Recall@k and MRR, scoring a hit when any chunk of the gold document appears."""
    hits, reciprocal, misses = 0, 0.0, []
    for query, gold in queries:
        found = [c.doc_id for c, _ in index.search(query, k=k, dedupe_docs=True)]
        if gold in found:
            hits += 1
            reciprocal += 1.0 / (found.index(gold) + 1)
        else:
            misses.append((query, gold, found[:3]))
    n = len(queries)
    return {"recall": hits / n, "mrr": reciprocal / n, "misses": misses}


def report(name, index, k=5):
    row = {"name": name}
    for group, queries in GROUPS.items():
        row[group] = evaluate(index, queries, k)["mrr"]
    overall = evaluate(index, ALL, k)
    row["all_mrr"] = overall["mrr"]
    row["all_recall"] = overall["recall"]
    row["misses"] = overall["misses"]
    return row


def print_table(rows, k):
    header = f"{'method':<30}{'semantic':>10}{'lexical':>9}{'mixed':>8}{'ALL mrr':>10}{f'  r@{k}':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<30}{r['semantic']:>10.3f}{r['lexical']:>9.3f}"
              f"{r['mixed']:>8.3f}{r['all_mrr']:>10.3f}{r['all_recall']:>8.3f}")


def main():
    k = 5
    embedder = Embedder()

    print(f"{len(DOCS)} documents, {len(ALL)} labelled queries, k={k}")
    print("columns are MRR within each query type; higher is better.\n")

    t0 = time.time()
    configs = [
        ("BM25 only", dict(mode="lexical")),
        ("dense only", dict(mode="dense")),
        ("hybrid, RRF", dict(mode="hybrid", fusion="rrf")),
        ("hybrid, linear a=0.5", dict(mode="hybrid", fusion="linear", alpha=0.5)),
        ("hybrid, linear a=0.7", dict(mode="hybrid", fusion="linear", alpha=0.7)),
    ]
    rows = [report(name, HybridIndex(embedder=embedder, **kw).build(DOCS), k)
            for name, kw in configs]
    print("=== retrieval method (chunking held at sentence-450) ===\n")
    print_table(rows, k)

    print("\n=== chunking strategy (retrieval held at hybrid RRF) ===\n")
    chunk_rows = [
        report(strategy,
               HybridIndex(mode="hybrid", fusion="rrf", strategy=strategy,
                           embedder=embedder).build(DOCS), k)
        for strategy in ("whole", "paragraph", "fixed-400",
                         "fixed-400-overlap", "sentence-450")
    ]
    print_table(chunk_rows, k)

    best = max(rows + chunk_rows, key=lambda r: r["all_mrr"])
    print(f"\nbest overall: {best['name']}  (MRR {best['all_mrr']:.3f})")
    print(f"elapsed: {time.time() - t0:.1f}s")

    if best["misses"]:
        print("\nqueries the best config still gets wrong:")
        for query, gold, got in best["misses"]:
            print(f"  {query!r}\n    want {gold}, got {got}")


if __name__ == "__main__":
    sys.exit(main())
# final output is a table of MRR scores for each retrieval method and chunking strategy, along with the best overall configuration and any queries it still misses.