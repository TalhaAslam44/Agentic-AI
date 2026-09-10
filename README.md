# Agentic-AI

A ladder of AI projects, built one at a time, from first principles up to a
deployable service. See [ROADMAP.md](ROADMAP.md) for the full plan.

| # | Project | Status |
|---|---------|--------|
| 1 | [Semantic search from scratch](01-semantic-search/) | done |
| 2 | [Neural embeddings, chunking, hybrid retrieval](02-embeddings-hybrid/) | done |
| 3 | RAG question answering | next |
| 4 | Tool-using agent | |
| 5 | Tabular ML pipeline | |
| 6 | Vision classifier + deployment | |
| 7 | Production RAG service | |

## Running a project

Each folder is self-contained with its own README.

```bash
cd 01-semantic-search
python evaluate.py
```

Project 1 needs only numpy. Project 2 onward use the repo's `.venv`:

```bash
.venv\Scripts\activate
```
