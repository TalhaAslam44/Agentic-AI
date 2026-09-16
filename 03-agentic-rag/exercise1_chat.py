"""Exercise 1: build a chatbot that remembers. YOU write this one.

Run from inside the 03-agentic-rag folder:
    cd 03-agentic-rag
    python exercise1_chat.py

Goal
----
A loop in the terminal. You type, Gemini answers, and Gemini remembers what
you said earlier in the conversation.

Test that memory works:
    you> my name is Talha
    you> what is my name?
If the second answer does not know your name, your history is broken.

The key fact you need
---------------------
The API remembers nothing. Memory is a Python list that YOU keep, append to,
and resend in full on every call. Step 1 showed a list with one message.
Here the list grows.

Fill in the four TODOs. Nothing else needs changing.
"""

from google import genai
from google.genai import types

from config import MODEL

client = genai.Client()
SYSTEM = "You are a friendly tutor. Keep answers short."


def ask(history):
    """Send the whole history, return (reply_text, usage). Already written for you."""
    response = client.models.generate_content(
        model=MODEL,
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            max_output_tokens=4000,
            # Turn off the SDK's automatic tool running. We will run tools ourselves
            # in Step 6, because writing that loop by hand is the whole lesson.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    parts = response.candidates[0].content.parts or []
    text = "".join(p.text for p in parts if p.text and not p.thought)
    return text, response.usage_metadata


def main():
    
    history = []

    total_input_tokens = 0
    
    while True:
        user_text = input("\nyou> ").strip()
        if not user_text:
            break

        history.append({"role": "user", "parts": [{"text": user_text}]})

        reply, usage = ask(history)
        print(f"gemini> {reply}")

        history.append({"role": "model", "parts": [{"text": reply}]})

        total_input_tokens += usage.input_tokens

    print(f"\nconversation over. total input tokens: {total_input_tokens}")


if __name__ == "__main__":
    main()
