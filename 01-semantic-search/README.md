# Project 1 — Semantic search from scratch

A working search engine in ~200 lines of numpy. No frameworks, no API keys, no
downloads. The goal is to rebuild the intuition that everything in modern
retrieval rests on: **a document is a vector, and relevance is an angle.**

## Run it

```bash
python evaluate.py
python search.py "why does my model memorise the training set"
python search.py            # interactive
```

## Files

| File | What it teaches |
|------|-----------------|
| `vectorize.py` | Tokenizing, TF-IDF weighting, L2 normalization, truncated SVD for latent semantics |
| `index.py` | The vector-store interface: build, embed a query, score, take top-k |
| `evaluate.py` | Labelled queries, recall@k and MRR, comparing configurations honestly |
| `search.py` | A thin CLI over the index |
| `corpus.py` | 40 short documents across science, ML, systems, history and economics |

## The three ideas worth absorbing

**1. Normalize, then dot.** Once every vector has unit length, a dot product
*is* cosine similarity. That single trick turns ranking into one matrix
multiply, and it is why vector databases are fast.

**2. TF-IDF cannot generalize.** It matches literal tokens. LSA projects the
term-document matrix onto its top singular directions, so words that co-occur
start to overlap. It is a weak, purely statistical version of what a neural
embedding does — but the plumbing around it is identical, which is why swapping
in real embeddings in Project 2 changes almost no code.

**3. You do not have a search engine until you can measure it.** `evaluate.py`
has 40 queries written in *different words* than the documents, and it prints
recall@5 and MRR for six configurations. That table is the deliverable, not the
demo.

## Current results

```
mode                      recall@5     MRR
tfidf                        0.975   0.913
lsa (k=32)                   0.975   0.930
lsa (k=64)                   0.975   0.913
hybrid a=0.5, k=64           0.975   0.913
```

Note that LSA barely beats plain TF-IDF here. That is an honest result, not a
bug: with 40 documents there is not enough co-occurrence data for SVD to learn
much. Reporting it anyway is the habit to build.

## The one failure, and why it matters

`"turning a hostname into an address"` never finds the DNS document, which says
`"translates human readable names into IP addresses"`.

- `hostname` appears nowhere in the corpus.
- `address` does not match `addresses`, because there is **no stemming**.

So every score is zero and the ranking is arbitrary. This is the exact failure
mode that killed keyword search and created the market for embeddings.

## Stretch tasks — do these before moving on

1. **Add a stemmer.** A crude suffix stripper (`-s`, `-es`, `-ing`, `-ed`) is
   enough. Re-run `evaluate.py`. Does MRR go up? Does anything regress?
2. **Handle the zero-score case.** When every score rounds to zero, return no
   results instead of three arbitrary documents. Silent garbage is worse than an
   empty list.
3. **Add character n-grams** (3–4 chars) alongside word features so `hostname`
   partially matches `host` and `name`.
4. **Sweep `alpha` and `n_components`** properly. Print a grid. Watch yourself
   start to overfit 40 queries, and think about what an honest split would be.
5. **Replace the corpus** with something you actually own — your notes, a repo's
   markdown files, saved articles. Write ten real queries for it. Real data
   breaks assumptions that toy data hides.
6. **Swap cosine for dot product on unnormalized vectors.** Watch long documents
   win everything, and understand why normalization was not optional.

## Where this goes next

Project 2 keeps `index.py`'s interface and replaces the vectorizer with a real
embedding model plus BM25, then puts it behind an approximate-nearest-neighbour
index. Project 3 feeds the retrieved documents to Claude and makes it cite them.
