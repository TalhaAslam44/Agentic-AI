"""Step 0: check your key works, and see which models you can call.

Run from inside the 03-agentic-rag folder:
    cd 03-agentic-rag
    python step0_list_models.py

This call is free. It lists models without generating any text.

A name appearing here does not guarantee you can use it. Retired models can
still be listed, and calling one returns a 404 that names its replacement.
"""

from google import genai

from config import MODEL

# The client reads your key from the GEMINI_API_KEY environment variable.
client = genai.Client()

names = []
for model in client.models.list():
    # Only keep models that can generate text. The list also includes
    # embedding models and others that cannot answer a question.
    if "generateContent" in (model.supported_actions or []):
        names.append(model.name.removeprefix("models/"))

print(f"{len(names)} text models available to your key:\n")
for name in sorted(names):
    marker = "   <-- config.py uses this" if name == MODEL else ""
    print(f"  {name}{marker}")

if MODEL not in names:
    print(f"\nWARNING: config.py uses '{MODEL}', which is not in this list.")
    print("Pick a 'flash' model from above and set it in config.py.")

#models picking up the most recent models from the list above, as of June 2024:
#  - Gemini 1.5 (flash)  --> "gemini-1.5-flash"