"""Step 1: talk to a language model, and look at everything that comes back.

Run:  python step1_first_call.py

Read this file top to bottom before running it. Every line is here on purpose.
"""

import anthropic

# ---------------------------------------------------------------------------
# 1. The client
# ---------------------------------------------------------------------------
# The client reads your key from the ANTHROPIC_API_KEY environment variable.
# Never paste a key into source code: this repo is on GitHub, and bots scan
# public repos for leaked keys within minutes.
client = anthropic.Anthropic()

MODEL = "claude-opus-5"

# ---------------------------------------------------------------------------
# 2. The request
# ---------------------------------------------------------------------------
# A language model API is stateless. The model remembers nothing between calls.
# Every request carries the ENTIRE conversation so far, as a list of messages.
# "Memory" in a chatbot is just your code resending the history each time.
#
#   system    who the model is and the rules it follows. Set by you, the builder.
#   messages  the conversation. Roles alternate "user" and "assistant".
#   max_tokens  a hard ceiling on the length of the reply. Hit it and the reply
#               is cut off mid-sentence, so do not set it low to save money.
response = client.beta.messages.create(
    model=MODEL,
    max_tokens=2000,
    system="You are a concise tutor. Answer in at most three sentences.",
    messages=[
        {"role": "user", "content": "What is retrieval augmented generation?"},
    ],
    # If a safety check declines the request, rerun it on a fallback model
    # instead of failing. You can ignore these two lines while learning.
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",
)

# ---------------------------------------------------------------------------
# 3. The response: check WHY it stopped before trusting WHAT it said
# ---------------------------------------------------------------------------
# stop_reason is the single most important field in agentic code.
#   end_turn    the model finished its answer normally
#   max_tokens  it ran out of room; the answer is truncated
#   tool_use    it wants YOU to run a tool (Step 5, this is where agents begin)
#   refusal     it declined the request
print("stop_reason:", response.stop_reason)
if response.stop_reason == "max_tokens":
    print("WARNING: reply was cut off, raise max_tokens")

# content is a LIST of blocks, not a string. A reply can contain several kinds
# of block: text, thinking, and later tool_use. Always check the type.
print("\nblocks returned:", [block.type for block in response.content])

answer = "".join(block.text for block in response.content if block.type == "text")
print("\nanswer:\n" + answer)

# ---------------------------------------------------------------------------
# 4. Usage: you pay per token, in and out
# ---------------------------------------------------------------------------
# Input tokens are everything you sent: system prompt plus full history.
# In a long chat or an agent loop, input grows every turn because the whole
# history is resent. That is the hidden cost curve of agents.
usage = response.usage
print(f"\ninput tokens:  {usage.input_tokens}")
print(f"output tokens: {usage.output_tokens}")

# Opus 5 pricing: $5 per million input tokens, $25 per million output tokens.
cost = usage.input_tokens * 5 / 1_000_000 + usage.output_tokens * 25 / 1_000_000
print(f"cost of this call: ${cost:.5f}")
