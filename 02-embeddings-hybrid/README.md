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

## What the numbers actually showed

Retrieval method, chunking held at sentence-450:

```
method                          semantic  lexical   mixed   ALL mrr     r@5
BM25 only                          0.538    1.000   1.000     0.830   0.947
dense only                         0.774    0.903   0.958     0.873   0.974
hybrid, RRF                        0.673    0.958   1.000     0.866   1.000
hybrid, linear a=0.5               0.714    1.000   0.958     0.882   1.000
hybrid, linear a=0.7               0.804    1.000   0.958     0.914   1.000
```

Best combination found, sweeping chunking and fusion together:

```
fixed-400 + linear a=0.7           0.863    1.000   1.000     0.950   1.000
fixed-400-overlap + linear a=0.7   0.848    1.000   1.000     0.944   1.000
paragraph + linear a=0.7           0.833    1.000   1.000     0.939   0.974
sentence-450 + RRF                 0.673    0.958   1.000     0.866   1.000
```

Four things here are worth more than the headline number.

**BM25's semantic column is the whole reason this project exists.** It scores
0.538 on paraphrases against 1.000 on jargon. Dense retrieval nearly closes that
gap, reaching 0.774, and the best hybrid reaches 0.863.

**Reciprocal rank fusion lost to linear fusion, badly.** RRF at 0.866 is *worse
than dense alone* at 0.873. RRF throws away score magnitudes and keeps only
ranks, which is exactly what protects it from scale mismatch in production, but
that discarding costs real information. With only 12 documents, BM25 confidently
ranking something first counts the same as it barely preferring it. The usual
production default is not the best choice at this scale, and only measurement
revealed that.

**The best chunking strategy was the ugly one.** Fixed 400-character windows,
which cut sentences in half, beat careful sentence-aware chunking. Plausibly the
smaller, more uniform chunks give sharper embeddings, and cutting mid-sentence
hurts a human reader far more than it hurts a cosine similarity. If these chunks
were being shown to a user as citations, the ranking might well reverse.

**Recall@5 hit 1.000 for every hybrid configuration.** That metric is saturated
and now useless here. With 12 documents and k=5, finding the right one in the
top five is too easy. Only MRR still discriminates. A metric that cannot
distinguish your options has stopped doing its job.

## The honest caveat

Every number above was measured on the same 38 queries that guided every choice
of chunk size, fusion method and alpha. That set has been fitted to. The gain
from 0.866 to 0.950 is partly real and partly the tuning showing up in its own
scorecard, and there is no way to tell the two apart from this table alone.

`doc-evaluation` in the corpus warns about exactly this, which is a convenient
irony. The fix is a held-out set of queries written before tuning and looked at
once at the end. Stretch task 7 is where you build one.

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
