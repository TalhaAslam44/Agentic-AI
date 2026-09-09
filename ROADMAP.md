# AI Project Ladder

Each project is standalone but reuses ideas from the last one. Build in order.

| # | Project | New skills | Stack |
|---|---------|-----------|-------|
| 1 | Semantic search from scratch | Embeddings, cosine similarity, TF-IDF/LSA, retrieval eval (recall@k, MRR) | numpy only |
| 2 | Real embeddings + vector DB | Neural embeddings, chunking strategy, ANN index, hybrid search (BM25 + dense) | sentence-transformers / API, FAISS or Chroma |
| 3 | RAG question answering | Prompt construction, context packing, citations, hallucination eval, LLM-as-judge | Claude API |
| 4 | Tool-using agent | Function calling, agent loop, error recovery, tracing, cost/latency budgets | Claude API tool use |
| 5 | Tabular ML done properly | Leakage-free pipelines, CV, calibration, feature importance, model cards | sklearn, pandas |
| 6 | Vision classifier + deployment | Transfer learning, augmentation, ONNX export, inference API | torch, ONNX, FastAPI |
| 7 | Production RAG service | Auth, caching, streaming, observability, evals in CI, Docker | FastAPI, Docker, pytest |

## How to work through them
1. Read the project README and try to write it yourself first.
2. Compare against the reference implementation.
3. Do the "stretch" tasks at the bottom of each README. They are where the learning is.
