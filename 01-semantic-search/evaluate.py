"""Retrieval evaluation. Without this you are guessing, not engineering."""

import numpy as np
from corpus import DOCS
from index import VectorIndex

# Queries phrased the way a person would ask, not by copying document words.
# That mismatch is exactly what a semantic index is supposed to bridge.
QUERIES = [
    ("how do plants make food from light", "d01"),
    ("where does a cell get its energy", "d02"),
    ("why does it rain", "d03"),
    ("what causes earthquakes", "d04"),
    ("molten rock erupting from a mountain", "d05"),
    ("how does an optimizer minimise loss", "d06"),
    ("my model does great on training data and badly on test data", "d07"),
    ("k fold estimate of generalisation error", "d08"),
    ("backpropagation through layers of weights", "d09"),
    ("dense vectors that capture word meaning", "d10"),
    ("self attention replaced recurrence", "d11"),
    ("byte pair encoding subword vocabulary", "d12"),
    ("reliable ordered delivery of packets", "d13"),
    ("turning a hostname into an address", "d14"),
    ("headers that let a proxy reuse a response", "d15"),
    ("avoid scanning every row in a table", "d16"),
    ("all or nothing group of statements", "d17"),
    ("removing redundant columns from tables", "d18"),
    ("automatic memory reclamation", "d19"),
    ("running tasks at the same time on many cores", "d20"),
    ("reviewing and merging code changes", "d21"),
    ("automated build and test on every push", "d22"),
    ("packaging an app with its dependencies", "d23"),
    ("encrypt with one key decrypt with another", "d24"),
    ("storing credentials safely with salt", "d25"),
    ("certificate verification before sending data", "d26"),
    ("senate and elected magistrates in ancient rome", "d27"),
    ("steam factories and railways changed cities", "d28"),
    ("movable type spread books and literacy", "d29"),
    ("bacteria that no longer respond to drugs", "d30"),
    ("training the immune system before infection", "d31"),
    ("three billion base pairs of dna", "d32"),
    ("gravity so strong light cannot escape", "d33"),
    ("galaxies moving away from us", "d34"),
    ("why a satellite does not fall down", "d35"),
    ("price where buyers and sellers agree", "d36"),
    ("money buys less than it used to", "d37"),
    ("interest earning interest over time", "d38"),
    ("merge sort and quicksort running time", "d39"),
    ("how runtime grows with input size", "d40"),
]


def evaluate(index, queries=QUERIES, k=5):
    """Recall@k = fraction of queries whose gold doc appears in the top k.
    MRR    = mean of 1/rank of the gold doc, so rank 1 counts far more than 5."""
    hits, reciprocal = 0, 0.0
    misses = []
    for query, gold in queries:
        results = index.search(query, k=k)
        ranked = [r[0] for r in results]
        if gold in ranked:
            hits += 1
            reciprocal += 1.0 / (ranked.index(gold) + 1)
        else:
            misses.append((query, gold, ranked[:3]))
    n = len(queries)
    return {"recall@k": hits / n, "mrr": reciprocal / n, "misses": misses}


def main():
    print(f"{len(DOCS)} documents, {len(QUERIES)} labelled queries\n")
    print(f"{'mode':<24}{'recall@5':>10}{'MRR':>8}")
    print("-" * 42)

    configs = [
        ("tfidf", dict(mode="tfidf")),
        ("lsa (k=32)", dict(mode="lsa", n_components=32)),
        ("lsa (k=64)", dict(mode="lsa", n_components=64)),
        ("hybrid a=0.7, k=64", dict(mode="hybrid", n_components=64, alpha=0.7)),
        ("hybrid a=0.5, k=64", dict(mode="hybrid", n_components=64, alpha=0.5)),
        ("hybrid a=0.3, k=64", dict(mode="hybrid", n_components=64, alpha=0.3)),
    ]

    best = None
    for name, kwargs in configs:
        idx = VectorIndex(**kwargs).build(DOCS)
        m = evaluate(idx)
        print(f"{name:<24}{m['recall@k']:>10.3f}{m['mrr']:>8.3f}")
        if best is None or m["mrr"] > best[1]["mrr"]:
            best = (name, m)

    print(f"\nbest: {best[0]}")
    if best[1]["misses"]:
        print("\nqueries it still gets wrong:")
        for query, gold, got in best[1]["misses"]:
            print(f"  {query!r}\n    want {gold}, got {got}")


if __name__ == "__main__":
    main()
