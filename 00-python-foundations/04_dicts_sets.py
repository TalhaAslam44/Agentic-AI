"""
LESSON 04: Dictionaries and Sets
==================================
Run:  python 00-python-foundations/04_dicts_sets.py

What you'll learn:
  1. Dicts: key -> value mapping (the shape of JSON)
  2. Safe access, updating, looping
  3. How dicts work inside: hash tables and why keys must be hashable
  4. Nested dicts (real API responses)
  5. Sets: uniqueness and fast membership
  6. collections: Counter, defaultdict
"""
from collections import Counter, defaultdict
import json

print("=" * 60)
print("1. DICTIONARY BASICS")
print("=" * 60)

config = {
    "model": "claude-sonnet-5",
    "max_tokens": 1024,
    "temperature": 0.2,
}
print(config["model"])
config["temperature"] = 0.5  # update
config["stream"] = True  # add new key
del config["stream"]  # delete
print(config)
print(len(config), "model" in config, "top_p" in config)  # `in` checks KEYS

print()
print("=" * 60)
print("2. SAFE ACCESS & LOOPING")
print("=" * 60)

# config["top_p"]  -> KeyError!
print(config.get("top_p"))  # None
print(config.get("top_p", 1.0))  # default value
print(config.setdefault("top_k", 40))  # get it, or set it if it's missing
removed = config.pop("top_k")
print("popped:", removed)

for key in config:  # loops over keys
    print("key:", key)
for key, value in config.items():  # the most common pattern
    print(f"{key:>12} = {value}")
print(list(config.keys()), list(config.values()))

# Merging (Python 3.9+)
defaults = {"temperature": 1.0, "max_tokens": 256, "top_p": 1.0}
overrides = {"temperature": 0.2}
final = defaults | overrides  # right side wins
print(final)

# Dict comprehension (more in lesson 05)
word_lengths = {w: len(w) for w in ["agent", "tool", "memory"]}
print(word_lengths)

# Dicts keep insertion order (guaranteed since Python 3.7)

print()
print("=" * 60)
print("3. UNDER THE HOOD: HASH TABLES")
print("=" * 60)

# dict lookup is O(1) on average, however big the dict is. How?
#   1. Python computes hash(key) -> a big integer
#   2. That integer picks a slot in an internal array
#   3. The value is stored/read at that slot
print(hash("model"), hash(42), hash((1, 2)))

# Keys must be HASHABLE, which means IMMUTABLE (their hash can never change).
ok = {(0, 0): "origin", "name": 1, 3.14: "pi"}
try:
    bad = {[1, 2]: "list key"}
except TypeError as e:
    print("TypeError:", e)  # unhashable type: 'list'

# Note: hash("model") changes between Python runs (hash randomization, a
# security feature). Never store hash() values in files.

print()
print("=" * 60)
print("4. NESTED DATA = JSON")
print("=" * 60)

# This is roughly what an LLM API response looks like after parsing JSON:
response = {
    "id": "msg_01",
    "model": "claude-sonnet-5",
    "content": [
        {"type": "text", "text": "Let me check the weather."},
        {"type": "tool_use", "name": "get_weather", "input": {"city": "Lahore"}},
    ],
    "usage": {"input_tokens": 52, "output_tokens": 31},
}

print(response["usage"]["output_tokens"])
for block in response["content"]:
    if block["type"] == "tool_use":
        print(f"Agent wants to call {block['name']} with {block['input']}")

total = sum(response["usage"].values())
print("total tokens:", total)

# dict <-> JSON string
as_text = json.dumps(response["usage"], indent=2)
print(as_text, type(as_text))
back = json.loads('{"city": "Karachi", "days": 3}')
print(back["city"], type(back))

print()
print("=" * 60)
print("5. SETS: unordered, unique, O(1) membership")
print("=" * 60)

tags = {"ml", "nlp", "ml", "agents"}  # duplicates removed
print(tags, len(tags))
empty = set()  # NOTE: {} is an empty DICT, not a set
tags.add("rag")
tags.discard("xyz")  # no error if it's missing (remove() would raise)

a = {"python", "sql", "pandas", "pytorch"}
b = {"python", "langchain", "pytorch", "docker"}
print("union       :", a | b)
print("intersection:", a & b)
print("difference  :", a - b)
print("sym diff    :", a ^ b)

# Deduplicate while keeping order (sets don't keep order, dicts do):
urls = ["x.com", "y.com", "x.com", "z.com", "y.com"]
print(list(dict.fromkeys(urls)))

# Speed: set membership vs list membership
import time

big_list = list(range(1_000_000))
big_set = set(big_list)
t = time.perf_counter(); 999_999 in big_list; t_list = time.perf_counter() - t
t = time.perf_counter(); 999_999 in big_set; t_set = time.perf_counter() - t
print(f"list lookup: {t_list*1e6:.0f} µs   set lookup: {t_set*1e6:.2f} µs")

print()
print("=" * 60)
print("6. COLLECTIONS HELPERS")
print("=" * 60)

text = "the agent calls the tool and the tool returns data"
counts = Counter(text.split())
print(counts.most_common(3))  # word frequency, which is how bag-of-words starts

by_label = defaultdict(list)  # missing key -> automatically becomes []
samples = [("spam", "win $$$"), ("ham", "meeting at 3"), ("spam", "free!!")]
for label, msg in samples:
    by_label[label].append(msg)
print(dict(by_label))

# ---------------------------------------------------------------
# AI CONNECTION
# ---------------------------------------------------------------
# - Tool definitions for agents are dicts (JSON Schema).
# - Tool arguments from the model arrive as JSON, so you parse them into a dict.
# - A tool registry is a dict mapping names to functions (see lesson 06).
# - Vocabulary / token-id mappings in NLP are dicts: {"hello": 15496}.

# ---------------------------------------------------------------
# EXERCISES (exercises/04.py)
# ---------------------------------------------------------------
# 1. From `response` above, collect the text of all "text" blocks into one string.
# 2. Count how many times each character appears in "mississippi" in two
#    ways: once with a plain dict and .get(), and once with Counter.
# 3. Given two lists of document IDs retrieved by two search methods, print
#    the IDs found by both methods and the IDs found by only one.
# 4. Invert a dict: {"a": 1, "b": 2} -> {1: "a", 2: "b"}.
# 5. Write a JSON string for a tool definition with name, description,
#    and a "parameters" dict, then json.loads it and print the tool name.
