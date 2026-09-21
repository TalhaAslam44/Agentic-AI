"""Exercise 2: a question-answering tool over your notes. YOU write the middle.

Run from inside the 03-agentic-rag folder:
    python exercise2_rag_cli.py

Goal
----
Ask a question, retrieve the best chunks, answer from them, and show which
chunks were used. Type a blank line to quit.

Everything except build_prompt() is written for you. That one function is the
heart of RAG, so you write it.

Try these questions once it runs:
    why is attention quadratic
    should chunks overlap
    what is mean reciprocal rank
    who won the world cup in 2018        <- nothing in the notes covers this
The last one matters. Watch what happens when retrieval finds nothing useful.
"""

from google import genai
from google.genai import types

from config import MODEL
from retrieval import search

client = genai.Client()

SYSTEM = """You are a research assistant answering questions about the user's notes.
Answer using the context provided in the user's message.
Keep answers under four sentences."""


def build_prompt(question, chunks):
    """YOU WRITE THIS.

    Inputs:
        question  a string
        chunks    a list of (chunk, score) pairs from search()
                  each chunk has .chunk_id (like "doc-rag#2") and .text

    Return one string containing:
        1. the text of every chunk, each labelled with its chunk_id
        2. the question, placed AFTER the context

    Hints:
        - Loop over chunks and collect strings in a list.
        - "\\n\\n".join(list_of_strings) glues them with blank lines between.
        - An f-string with triple quotes lets you write a multi-line template.
        - Look at build_prompt in step2_naive_rag.py only if you get stuck.
    """
    raise NotImplementedError("write build_prompt")


def ask(prompt):
    """Send one prompt, return (answer_text, usage). Written for you."""
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
    print("ask about the Project 2 notes. blank line to quit.")

    while True:
        question = input("\nquestion> ").strip()
        if not question:
            break

        chunks = search(question, k=3)

        print("\nretrieved:")
        for chunk, score in chunks:
            print(f"  [{score:.3f}] {chunk.chunk_id}")

        prompt = build_prompt(question, chunks)
        answer, usage = ask(prompt)

        print(f"\nanswer:\n{answer}")
        print(f"\n[input tokens: {usage.prompt_token_count}]")


if __name__ == "__main__":
    main()
