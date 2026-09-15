"""Exercise 1: build a chatbot that remembers. YOU write this one.

Run:  python exercise1_chat.py

Goal
----
A loop in the terminal. You type, Claude answers, and Claude remembers what
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

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"
SYSTEM = "You are a friendly tutor. Keep answers short."


def ask(history):
    """Send the whole history, return (reply_text, usage). Already written for you."""
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=history,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    return text, response.usage


def main():
    # TODO 1: create an empty list called history.

    total_input_tokens = 0

    while True:
        user_text = input("\nyou> ").strip()
        if not user_text:
            break

        # TODO 2: append the user's message to history.
        #         Shape: {"role": "user", "content": user_text}

        reply, usage = ask(history)
        print(f"claude> {reply}")

        # TODO 3: append Claude's reply to history, with role "assistant".
        #         If you skip this, what does Claude see on the next turn?

        # TODO 4: add usage.input_tokens to total_input_tokens, then print
        #         the turn number, this turn's input tokens, and the total.

    print(f"\nconversation over. total input tokens: {total_input_tokens}")


if __name__ == "__main__":
    main()
