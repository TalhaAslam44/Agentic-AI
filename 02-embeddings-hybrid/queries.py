"""Labelled evaluation queries.

Deliberately split into three groups, because a single average hides the whole
story. Lexical and dense retrieval fail on different queries, and the only way
to see that is to keep the groups separate in the report.
"""

# Paraphrases: the answer is stated in completely different words.
# Expect dense to win and BM25 to struggle.
SEMANTIC = [
    ("why do leaves look green", "doc-photosynthesis"),
    ("how does a plant survive in a desert", "doc-photosynthesis"),
    ("how does a model look at other words in a sentence", "doc-transformers"),
    ("why does a long input cost so much more to process", "doc-transformers"),
    ("turning a sentence into a list of numbers", "doc-embeddings"),
    ("why does the same word mean two different things", "doc-embeddings"),
    ("why cut a long article into pieces before storing it", "doc-chunking"),
    ("how big should the pieces be", "doc-chunking"),
    ("finding close neighbours without checking everything", "doc-vectordb"),
    ("how do I know if my search is any good", "doc-evaluation"),
    ("stopping a chatbot from making things up", "doc-rag"),
    ("teaching a model a new skill with a small dataset", "doc-finetuning"),
    ("getting the model to admit it does not know", "doc-prompting"),
    ("stopping a loop from running forever and costing money", "doc-agents"),
]

# Rare exact terms: jargon that appears verbatim in exactly one chunk.
# Expect BM25 to win and dense to be less reliable.
LEXICAL = [
    ("RuBisCO", "doc-photosynthesis"),
    ("rotary position embeddings", "doc-transformers"),
    ("flash attention", "doc-transformers"),
    ("word2vec and GloVe", "doc-embeddings"),
    ("k1 and b parameters", "doc-bm25"),
    ("HNSW", "doc-vectordb"),
    ("product quantisation", "doc-vectordb"),
    ("normalised discounted cumulative gain", "doc-evaluation"),
    ("cross encoder reranking", "doc-rag"),
    ("byte pair encoding", "doc-tokenization"),
    ("low rank adaptation", "doc-finetuning"),
    ("chain of thought", "doc-prompting"),
]

# Natural questions mixing both. This is what real traffic looks like.
MIXED = [
    ("what does the Calvin cycle do", "doc-photosynthesis"),
    ("why is attention quadratic", "doc-transformers"),
    ("what is cosine similarity used for", "doc-embeddings"),
    ("why does BM25 normalise document length", "doc-bm25"),
    ("should chunks overlap", "doc-chunking"),
    ("what does an inverted file index do", "doc-vectordb"),
    ("what is mean reciprocal rank", "doc-evaluation"),
    ("what are the stages of a RAG pipeline", "doc-rag"),
    ("why do tokens cost more in some languages", "doc-tokenization"),
    ("is fine tuning worth it", "doc-finetuning"),
    ("do few shot examples help", "doc-prompting"),
    ("how should a tool report a failure", "doc-agents"),
]

GROUPS = {"semantic": SEMANTIC, "lexical": LEXICAL, "mixed": MIXED}
ALL = SEMANTIC + LEXICAL + MIXED
