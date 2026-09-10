"""Splitting documents into retrievable units.

A whole document's embedding is the average of everything it says. For a
document covering five topics that average sits in the middle of all five and
is close to none of them. Chunking exists to give each vector one topic.
"""

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """One retrievable unit, with enough metadata to cite it later."""
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int   # index of this chunk within its document

    @property
    def display(self):
        """What actually gets embedded: the title gives an isolated chunk context."""
        return f"{self.title}: {self.text}"


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def chunk_whole(doc):
    """Baseline: one chunk per document. Shows why chunking is needed at all."""
    doc_id, title, text = doc
    return [Chunk(f"{doc_id}#0", doc_id, title, " ".join(text.split()), 0)]


def chunk_paragraphs(doc):
    """Split on blank lines. Respects the author's own topic boundaries."""
    doc_id, title, text = doc
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        Chunk(f"{doc_id}#{i}", doc_id, title, " ".join(p.split()), i)
        for i, p in enumerate(paras)
    ]


def chunk_fixed(doc, size=400, overlap=0):
    """Fixed character windows, ignoring all structure.

    Cheap and predictable, but it cuts sentences in half. Included so the
    evaluation can show what that costs rather than asserting it.
    """
    doc_id, title, text = doc
    text = " ".join(text.split())
    step = max(1, size - overlap)
    chunks = []
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc_id}#{i}", doc_id, title, piece.strip(), i))
        if start + size >= len(text):
            break
    return chunks


def chunk_sentences(doc, max_chars=450, overlap_sentences=1):
    """Pack whole sentences up to a size budget, overlapping at the seams.

    Every chunk is readable on its own, which matters when it is shown to a
    user as a citation or handed to a language model as context. The overlap
    means a fact spanning a boundary appears intact in at least one chunk.
    """
    doc_id, title, text = doc
    sentences = split_sentences(" ".join(text.split()))

    chunks, current, length, i = [], [], 0, 0
    for sentence in sentences:
        # Start a new chunk once adding this sentence would blow the budget,
        # but never emit an empty one just because a single sentence is long.
        if current and length + len(sentence) > max_chars:
            chunks.append(Chunk(f"{doc_id}#{i}", doc_id, title, " ".join(current), i))
            i += 1
            current = current[-overlap_sentences:] if overlap_sentences else []
            length = sum(len(s) for s in current)
        current.append(sentence)
        length += len(sentence)

    if current:
        chunks.append(Chunk(f"{doc_id}#{i}", doc_id, title, " ".join(current), i))
    return chunks


STRATEGIES = {
    "whole": chunk_whole,
    "paragraph": chunk_paragraphs,
    "fixed-400": lambda d: chunk_fixed(d, size=400, overlap=0),
    "fixed-400-overlap": lambda d: chunk_fixed(d, size=400, overlap=80),
    "sentence-450": lambda d: chunk_sentences(d, max_chars=450, overlap_sentences=1),
}


def build_chunks(docs, strategy="sentence-450"):
    fn = STRATEGIES[strategy]
    out = []
    for doc in docs:
        out.extend(fn(doc))
    return out


if __name__ == "__main__":
    from corpus import DOCS

    print(f"{'strategy':<22}{'chunks':>8}{'avg chars':>12}{'max chars':>12}")
    print("-" * 54)
    for name in STRATEGIES:
        chunks = build_chunks(DOCS, name)
        lengths = [len(c.text) for c in chunks]
        print(f"{name:<22}{len(chunks):>8}{sum(lengths)//len(lengths):>12}{max(lengths):>12}")
