# Project 2 — Neural embeddings, chunking and hybrid retrieval

Project 1 ended on a wall: two queries scored exactly zero because they shared
no vocabulary with the documents they should have found. This project knocks
that wall down three different ways, and measures each one.

Runs fully local. No API key, no cost.

## Setup

The repo has a `.venv` with torch and sentence-transformers installed.

```bash
.venv\Scripts\activate
```

In VS Code, press Ctrl+Shift+P, choose "Python: Select Interpreter", and pick
the one inside `.venv`.

## Run it

```bash
python evaluate.py
```

```bash
python search.py --compare "why do leaves look green"
```

```bash
python embedder.py
```

The first run downloads a 90MB model. After that, embeddings are cached to
`.cache/` and every run is instant.

## What is new since project 1

| Concept | Where |
|---|---|
| Chunking, four strategies compared | `chunking.py` |
| BM25 with saturation and length normalisation | `lexical.py` |
| Stemming, the fix for project 1's failures | `lexical.py` |
| Neural embeddings with a disk cache | `embedder.py` |
| Reciprocal rank fusion and linear fusion | `index.py` |
| Evaluation split by query type | `queries.py`, `evaluate.py` |

## The four ideas worth absorbing

**1. Chunking is not a detail.** A document covering five topics has an
embedding sitting in the middle of all five, close to none of them. Splitting it
so each vector holds one topic is usually a bigger win than upgrading the
embedding model. `chunking.py` runs standalone and prints the size distribution
of each strategy.

**2. BM25 fixes two lies in TF-IDF.** Term frequency saturates, so the twentieth
mention of a word barely beats the tenth. Document length is normalised, so a
long document cannot win just by containing more words. Two parameters control
these, `k1` and `b`, and the comments in `lexical.py` explain what each does at
its extremes.

**3. Lexical and dense retrieval fail on *different* queries.** This is the
entire argument for hybrid search, and it is why `queries.py` splits the
evaluation into three groups instead of reporting one average. Watch the
semantic column and the lexical column move in opposite directions as you change
methods. A single overall number would hide that completely.

**4. Fusing two rankers needs care.** BM25 scores are unbounded and cosine
similarity lives between -1 and 1, so adding the raw numbers lets BM25 drown the
dense side. Two honest fixes are implemented. Linear fusion squashes both to a
0-to-1 range first. Reciprocal rank fusion throws the scores away entirely and
combines only the *ranks*, which is why it needs no tuning and is the usual
production default.

## What the numbers show

Run `evaluate.py` for the live table. The shape to look for:

- BM25 scores near perfect on the rare-jargon queries and poorly on paraphrases.
- Dense does the reverse.
- Hybrid should beat both overall while giving up a little on each specialty.

If hybrid does *not* beat both, that is a real finding worth investigating, not
a reason to quietly drop the comparison.

## Stretch tasks

1. **Add a cross-encoder reranker.** Retrieve 20 candidates with hybrid, then
   rescore with `cross-encoder/ms-marco-MiniLM-L-6-v2` and keep the top 5. This
   is the single biggest quality jump available and it belongs in project 3.
2. **Sweep `rrf_k`.** Try 10, 60, 200. It controls how decisive first place is.
   Does the best value differ between query groups?
3. **Sweep `k1` and `b` in BM25.** Set `b=0` and watch long chunks take over.
4. **Turn stemming off** in `lexical.py` and re-run. Quantify what it bought.
5. **Make the dense side fail on purpose.** Add queries with product codes or
   version numbers. Confirm dense retrieval is worse than BM25 there, which is
   the failure mode people forget when they replace search with embeddings.
6. **Time the build.** Print how long embedding takes versus BM25 fitting, then
   think about what happens at a million chunks. That is where the vector
   database concepts in `doc-vectordb` stop being theory.
7. **Swap in your own documents.** Write 15 real queries against them. This is
   the only experiment that will tell you anything about *your* data.

## Where this goes next

Project 3 keeps this retriever, adds a cross-encoder reranker, and hands the
retrieved chunks to Claude with instructions to answer only from them and cite
which chunk supports each claim. The evaluation gains a second half: not just
whether the right chunk was found, but whether the answer was actually grounded
in it.
