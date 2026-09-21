"""Step 2: naive RAG. Retrieve first, then let the model answer.

Run from inside the 03-agentic-rag folder:
    python step2_naive_rag.py

This asks the same question twice: once with no help, and once with the three
best chunks from your Project 2 corpus pasted into the prompt. Compare them.
"""

from google import genai
from google.genai import types

from config import MODEL
from retrieval import search

client = genai.Client()

QUESTION = "According to my notes, what is the best chunk size to use?"

# ---------------------------------------------------------------------------
# The system instruction is where the rules live
# ---------------------------------------------------------------------------
# Step 3 will tighten these rules a lot. For now, just aim the model at the
# provided context.
SYSTEM = """You are a research assistant answering questions about the user's notes.
Answer using the context provided in the user's message.
Keep answers under four sentences."""


def build_prompt(question, chunks):
    """Turn retrieved chunks plus a question into one string for the model.

    This function IS the R and the A of RAG. Everything about it is a decision:
    the order of the chunks, the labels, where the question goes, and how
    clearly the context is separated from the instruction.
    """
    blocks = []
    for chunk, score in chunks:
        # Label every chunk with its id. The model cannot cite a source that
        # has no name, and in Step 3 you will require it to cite one.
        blocks.append(f"[{chunk.chunk_id}]\n{chunk.text}")

    context = "\n\n".join(blocks)

    # Context first, question last. Models attend strongly to the end of a
    # prompt, so the instruction you most want followed goes there.
    return f"""Here is the context from the user's notes:

<context>
{context}
</context>

Question: {question}"""


def ask(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=4000,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    parts = response.candidates[0].content.parts or []
    text = "".join(p.text for p in parts if p.text and not p.thought)
    return text, response.usage_metadata


def main():
    print(f"question: {QUESTION}\n")

    # -----------------------------------------------------------------------
    # 1. No retrieval. The model answers from training data alone.
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("WITHOUT CONTEXT")
    print("=" * 70)
    answer, usage = ask(QUESTION)
    print(answer)
    print(f"\n[input tokens: {usage.prompt_token_count}]")

    # -----------------------------------------------------------------------
    # 2. Retrieve, then answer. This is the whole of naive RAG.
    # -----------------------------------------------------------------------
    chunks = search(QUESTION, k=3)

    print("\n" + "=" * 70)
    print("RETRIEVED CHUNKS")
    print("=" * 70)
    for chunk, score in chunks:
        print(f"  [{score:.3f}] {chunk.chunk_id}")

    prompt = build_prompt(QUESTION, chunks)

    print("\n" + "=" * 70)
    print("WITH CONTEXT")
    print("=" * 70)
    answer, usage = ask(prompt)
    print(answer)
    print(f"\n[input tokens: {usage.prompt_token_count}]")

    # -----------------------------------------------------------------------
    # 3. Look at what you actually sent
    # -----------------------------------------------------------------------
    # Printing the final prompt is the single most useful debugging habit in
    # this whole field. Most "the model is stupid" bugs are really "the prompt
    # did not contain what I assumed it did".
    print("\n" + "=" * 70)
    print("THE PROMPT THAT WAS SENT")
    print("=" * 70)
    print(prompt)


if __name__ == "__main__":
    main()
