"""Step 1: talk to a language model, and look at everything that comes back.

Run from inside the 03-agentic-rag folder:
    cd 03-agentic-rag
    python step1_first_call.py

Read this file top to bottom before running it. Every line is here on purpose.
"""

from google import genai
from google.genai import types

from config import MODEL

# ---------------------------------------------------------------------------
# 1. The client
# ---------------------------------------------------------------------------
# The client reads your key from the GEMINI_API_KEY environment variable.
# Never paste a key into source code: this repo is on GitHub, and bots scan
# public repos for leaked keys within minutes.
client = genai.Client()

# ---------------------------------------------------------------------------
# 2. The request
# ---------------------------------------------------------------------------
# A language model API is stateless. The model remembers nothing between calls.
# Every request carries the ENTIRE conversation so far, as a list of messages.
# "Memory" in a chatbot is just your code resending the history each time.
#
#   system_instruction  who the model is and the rules it follows. Set by you.
#   contents            the conversation. Roles alternate "user" and "model".
#                       (Anthropic and OpenAI call the second role "assistant".
#                       Same idea, different word.)
#   max_output_tokens   a hard ceiling on the reply. Hit it and the reply is cut
#                       off. On thinking models, the hidden reasoning ALSO
#                       counts against it, so do not set it low.
response = client.models.generate_content(
    model=MODEL,
    contents=[
        {"role": "user", "parts": [{"text": "What is retrieval augmented generation?"}]},
    ],
    config=types.GenerateContentConfig(
        system_instruction="You are a concise tutor. Answer in at most three sentences.",
        max_output_tokens=4000,
        # Turn off the SDK's automatic tool running. We will run tools ourselves
        # in Step 6, because writing that loop by hand is the whole lesson.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    ),
)

# ---------------------------------------------------------------------------
# 3. The response: check WHY it stopped before trusting WHAT it said
# ---------------------------------------------------------------------------
# The response can hold several candidate answers. We asked for one.
candidate = response.candidates[0]

# finish_reason is the single most important field in agentic code.
#   STOP        the model finished its answer normally
#   MAX_TOKENS  it ran out of room; the answer is truncated or even empty
#   SAFETY      a safety filter blocked the answer
# In Step 5 you will also see the model ask YOU to run a tool instead of
# answering. That moment is where agents begin.
print("finish_reason:", candidate.finish_reason.name)
if candidate.finish_reason.name == "MAX_TOKENS":
    print("WARNING: reply was cut off, raise max_output_tokens")

# The answer is a list of PARTS, not a string. A reply can contain text parts,
# thought parts, and later function_call parts. Always check what each part is.
parts = candidate.content.parts or []
print("\nparts returned:", ["thought" if p.thought else "text" if p.text else "other" for p in parts])

answer = "".join(p.text for p in parts if p.text and not p.thought)
print("\nanswer:\n" + (answer or "(no text came back)"))

# ---------------------------------------------------------------------------
# 4. Usage: every token counts, in and out
# ---------------------------------------------------------------------------
# Input tokens are everything you sent: system instruction plus full history.
# In a long chat or an agent loop, input grows every turn because the whole
# history is resent. That is the hidden cost curve of agents.
#
# The free tier costs nothing, but it limits requests per minute and per day.
# Go over and you get a 429 error. Paid APIs bill by these same token counts.
usage = response.usage_metadata
print(f"\ninput tokens:     {usage.prompt_token_count}")
print(f"thinking tokens:  {usage.thoughts_token_count or 0}   (hidden reasoning)")
print(f"answer tokens:    {usage.candidates_token_count or 0}")
print(f"total tokens:     {usage.total_token_count}")
