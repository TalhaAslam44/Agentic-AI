"""Longer, multi-paragraph documents.

Project 1 used one-sentence documents, so chunking never mattered. These are
long enough that the answer to a question lives in one paragraph while the rest
of the document is noise. That is the situation chunking exists for.
"""

DOCS = [
    ("doc-photosynthesis", "Photosynthesis", """
Photosynthesis is the process by which plants, algae and some bacteria convert light energy into chemical energy. The overall reaction takes carbon dioxide and water and produces glucose and oxygen. It is the ultimate source of almost all the chemical energy used by life on Earth.

The light dependent reactions happen in the thylakoid membranes of the chloroplast. Chlorophyll absorbs photons, mostly in the blue and red parts of the spectrum, and the energy is used to split water molecules. This releases oxygen as a waste product and produces ATP and NADPH.

The Calvin cycle then runs in the stroma and does not directly require light. An enzyme called RuBisCO fixes carbon dioxide onto a five carbon sugar, and the resulting molecules are reduced using the ATP and NADPH from the first stage. RuBisCO is remarkably slow and is thought to be the most abundant protein on the planet.

Plants in hot dry climates evolved variations. C4 and CAM plants concentrate carbon dioxide before it reaches RuBisCO, which reduces wasteful photorespiration. Maize and sugarcane are C4 plants, while cacti and pineapples use CAM and open their stomata only at night.
"""),

    ("doc-transformers", "The transformer architecture", """
The transformer was introduced in 2017 and replaced recurrent networks as the default architecture for sequence modelling. Its central mechanism is self attention, which lets every position in a sequence look at every other position directly, rather than passing information along step by step.

Attention works by projecting each token into three vectors called the query, the key and the value. The query of one token is compared against the keys of all tokens by dot product, the scores are scaled and passed through a softmax, and the result is used to take a weighted average of the values. Because these are all matrix multiplications, the whole sequence is processed in parallel.

Multi head attention runs several of these operations side by side with different learned projections. Different heads specialise, with some tracking syntax and others tracking long range dependencies. The outputs are concatenated and projected back to the model dimension.

Since attention itself is order agnostic, position must be injected explicitly. The original paper used fixed sinusoidal encodings added to the input. Modern models mostly use rotary position embeddings, which rotate the query and key vectors by an angle proportional to position and generalise better to longer sequences.

The main cost is that attention is quadratic in sequence length, since every token attends to every other. Flash attention reduces the memory cost by never materialising the full attention matrix, and sparse or sliding window variants reduce the compute cost by restricting which positions can attend to which.
"""),

    ("doc-embeddings", "Text embeddings", """
An embedding maps a piece of text to a fixed length vector of floating point numbers, arranged so that texts with similar meanings land close together. The number of dimensions is typically a few hundred to a few thousand, and it does not depend on how long the input text was.

Early methods like word2vec and GloVe learned one vector per word from co-occurrence statistics. They famously captured analogies through vector arithmetic, but they gave every sense of a word the same vector, so the word bank near a river and the word bank holding money were identical.

Modern sentence embedding models are transformers fine tuned with a contrastive objective. The model sees pairs of texts that should be similar and batches of texts that should not, and it is trained to pull the matching pairs together and push everything else apart. The final vector is usually a mean pool over the token outputs.

Distance is almost always measured with cosine similarity, which cares only about direction and not magnitude. If the vectors are normalised to unit length first, cosine similarity is just a dot product, which is why vector search reduces to one large matrix multiplication.

Embeddings have real limits. They compress meaning lossily, so exact identifiers like product codes, error numbers or rare proper nouns are often matched poorly. They also inherit the biases and the vocabulary of their training data, and a model trained mostly on English web text will do badly on specialised legal or medical jargon.
"""),

    ("doc-bm25", "Lexical retrieval and BM25", """
Lexical retrieval scores documents by the query words they literally contain. Despite decades of work on semantic methods, a well tuned lexical scorer remains a very strong baseline and is still the first stage of most production search systems.

BM25 is the standard ranking function. It builds on term frequency and inverse document frequency but adds two important corrections. Term frequency saturates, so the twentieth occurrence of a word adds almost nothing over the tenth, controlled by a parameter usually called k1. Document length is normalised, so a long document does not win simply by containing more words, controlled by a parameter usually called b.

The inverse document frequency component uses a probabilistic form that can go negative for terms appearing in more than half the collection. Implementations usually floor it at a small positive value to avoid strange behaviour with very common words.

The great strength of lexical search is precision on rare exact terms. If a user searches for a specific error code, a part number or an unusual surname, BM25 will find it reliably while a dense embedding model may not. Its great weakness is that it has no notion of synonyms, so a query and a document that mean the same thing in different words score zero against each other.

This complementary pattern is why hybrid retrieval works. Lexical and dense methods fail on different queries, so combining them recovers results that either one alone would miss.
"""),

    ("doc-chunking", "Chunking strategies", """
Retrieval systems rarely index whole documents. A long document covers many topics, and its average embedding is a blurry mixture that matches nothing well. Splitting documents into smaller chunks gives each vector a single coherent topic.

Fixed size chunking splits on a token or character count. It is trivial to implement and gives predictable memory use, but it cuts sentences in half and separates a claim from the evidence supporting it.

Sentence aware and paragraph aware chunking respects natural boundaries. Chunks vary in length but each one is readable on its own, which matters a great deal when the chunk is later shown to a language model or to a user as a citation.

Overlapping chunks repeat some content between neighbours, typically ten to twenty percent. The overlap means a fact sitting near a boundary appears whole in at least one chunk. The cost is a larger index and duplicate results that need to be deduplicated after retrieval.

Chunk size is a genuine tradeoff rather than a solved problem. Small chunks give precise matching but may lack the surrounding context needed to be useful. Large chunks carry context but dilute the embedding and waste space in a model's context window. The only reliable way to choose is to measure retrieval quality on your own queries.
"""),

    ("doc-vectordb", "Vector databases and approximate search", """
Exact nearest neighbour search compares the query against every stored vector. This is a single matrix multiplication and is perfectly fast for tens of thousands of vectors, but the cost grows linearly and becomes impractical at tens of millions.

Approximate nearest neighbour indexes trade a small amount of recall for a very large speedup. The dominant method is HNSW, which builds a layered graph where each layer is a navigable small world network. Search starts at a sparse top layer, greedily walks toward the query, then descends and refines.

Inverted file indexes take a different approach. Vectors are clustered, the query is compared only against the nearest few clusters, and everything else is skipped. This is often combined with product quantisation, which compresses each vector into a short code by splitting it into subvectors and replacing each with a codebook entry.

Every approximate index has a knob that trades recall against latency. In HNSW it is the size of the candidate list during search. Tuning it requires measuring recall against exact search on real queries, because the right setting depends entirely on how tolerant your application is to a missed result.

Metadata filtering complicates all of this. Restricting a search by date, owner or category interacts badly with graph traversal, since the reachable neighbours may all be filtered out. Systems solve this with pre filtering, post filtering or specialised filtered graph traversal, and the correct choice depends on how selective the filter is.
"""),

    ("doc-evaluation", "Evaluating retrieval", """
A retrieval system cannot be improved without measurement, and the measurement needs labelled queries where the correct documents are known. Building this set is the least glamorous and most valuable part of the work.

Recall at k asks whether the correct document appears anywhere in the top k results. It is the right metric when a downstream stage, such as a reranker or a language model, will read all k results and sort out which matters.

Mean reciprocal rank averages one divided by the rank of the first correct result. It rewards putting the answer at the very top, which matters when a user only looks at the first result or two.

Normalised discounted cumulative gain handles the case where relevance is graded rather than binary and several documents are relevant to different degrees. It discounts gains logarithmically by rank and normalises against the best possible ordering.

The most common mistake is tuning parameters until the numbers on the evaluation set look good, then reporting those numbers as if they predicted real performance. Once a set has guided dozens of decisions it has been fitted to, and an honest system keeps a held out set that is only looked at rarely.
"""),

    ("doc-rag", "Retrieval augmented generation", """
Retrieval augmented generation gives a language model access to information it was not trained on by retrieving relevant text at query time and placing it in the prompt. This keeps answers current without retraining and lets the model cite sources.

A basic pipeline has four stages. Documents are chunked and embedded once, offline. At query time the question is embedded and used to retrieve the closest chunks. Those chunks are formatted into a prompt with instructions. The model then generates an answer grounded in what it was given.

Retrieval quality dominates the final answer quality. If the correct chunk is not retrieved, no amount of prompt engineering will recover it, and the model will either refuse or invent something plausible. Most disappointing systems are failing at retrieval while their owners are tuning the prompt.

Reranking is the usual next improvement. A cheap retriever fetches perhaps fifty candidates and a cross encoder, which reads the query and the document together rather than embedding them separately, rescores them and keeps the best handful. Cross encoders are far more accurate and far too slow to run over an entire collection.

Grounding must be verified rather than assumed. Asking the model to cite the chunk supporting each claim makes fabrication visible, and a separate check that every citation actually contains the claim catches the cases where the model cites a source that does not support what it said.
"""),

    ("doc-tokenization", "Tokenization", """
Language models do not read characters or words. Text is first converted into tokens, which are integer identifiers drawn from a fixed vocabulary, usually somewhere between thirty thousand and two hundred thousand entries.

Byte pair encoding starts from individual bytes and repeatedly merges the most frequent adjacent pair, adding each merged unit to the vocabulary. Common words end up as a single token while rare words are assembled from several pieces. Working at the byte level guarantees that any input can be encoded, with no unknown token.

Tokenization has practical consequences that surprise people. The same text costs different numbers of tokens in different languages, with English typically cheapest and languages written in non Latin scripts often several times more expensive. Numbers are split inconsistently, which is part of why models struggle with arithmetic. Whitespace usually attaches to the following word, so a leading space changes the token.

Because billing, context limits and truncation all operate on tokens rather than characters, any system that packs retrieved text into a prompt has to count tokens with the model's actual tokenizer rather than estimating from character counts.
"""),

    ("doc-finetuning", "Fine tuning and adaptation", """
Fine tuning continues training a pretrained model on a smaller task specific dataset. It changes the model's behaviour in a way that prompting cannot, but it needs labelled data, compute, and a way to evaluate whether the result actually improved.

Full fine tuning updates every parameter and requires memory for the weights, the gradients and the optimiser state, which is several times the size of the model. Parameter efficient methods avoid this. Low rank adaptation freezes the original weights and learns small low rank update matrices, cutting trainable parameters by orders of magnitude while matching full fine tuning on many tasks.

Quantised variants push this further by holding the frozen base model in four bit precision while training the adapters in higher precision, which brings fine tuning of large models onto a single consumer graphics card.

The honest first question is whether fine tuning is needed at all. Prompting, few shot examples and retrieval solve a large fraction of problems more cheaply and are far easier to change later. Fine tuning earns its place when you need a consistent output format, a specialised domain vocabulary, or lower latency from a smaller model.

Catastrophic forgetting is the main risk. A model trained hard on a narrow task loses general capability, so evaluation has to cover both the target task and a sample of general behaviour.
"""),

    ("doc-prompting", "Prompting techniques", """
A prompt is the entire input a language model conditions on, and small changes to it produce large changes in output. Treating prompts as code, with version control and tests, is the difference between a demo and a system.

Few shot prompting includes worked examples in the input. The model infers the pattern and the format from the examples without any weight updates. Consistency of formatting between the examples matters more than the number of them.

Chain of thought prompting asks the model to reason step by step before answering. It improves accuracy on multi step arithmetic and logic problems, at the cost of more output tokens and higher latency. Reasoning models do this internally and generally do not need to be asked.

Structure helps a great deal. Putting instructions and data in clearly delimited sections, stating the output format explicitly, and telling the model what to do when it cannot answer all reduce the variance of the response.

The most reliable single improvement is giving the model an explicit escape route. A model told to answer from the provided context and to say it does not know when the context is insufficient will fabricate far less than one simply told to answer the question.
"""),

    ("doc-agents", "Tool using agents", """
An agent is a language model placed in a loop with tools. The model receives a goal, decides which tool to call, sees the result, and repeats until it decides the task is done. The tools are ordinary functions described to the model with a name, a description and a schema for the arguments.

The description of a tool is part of the prompt and deserves the same care. Ambiguous parameter names and vague descriptions cause the model to call the wrong tool or to pass badly formed arguments far more often than the underlying model capability would suggest.

Error handling separates working agents from fragile ones. When a tool fails, returning a clear error message describing what went wrong lets the model correct itself, whereas raising an exception or returning an empty result usually leads to the same failed call being repeated.

Loops need explicit limits. A maximum number of iterations, a token budget and a wall clock timeout are all necessary, because a confused agent will otherwise retry indefinitely and produce a surprising bill.

Observability is not optional. Recording every prompt, tool call, result and timing is the only practical way to understand why an agent behaved as it did, since the same input can produce different trajectories on different runs.
"""),
]
