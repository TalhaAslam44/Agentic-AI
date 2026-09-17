"""
LESSON 05: Control Flow (conditions, loops, comprehensions)
============================================================
Run:  python 00-python-foundations/05_control_flow.py

What you'll learn:
  1. if / elif / else, truthiness, and/or short-circuiting
  2. match (structural pattern matching)
  3. for loops, range, how iteration works inside
  4. while loops, break, continue, for-else
  5. Comprehensions (list/dict/set) and generator expressions
  6. A mini agent loop that puts it all together
"""

print("=" * 60)
print("1. CONDITIONS")
print("=" * 60)

confidence = 0.64
if confidence >= 0.8:
    decision = "answer"
elif confidence >= 0.5:
    decision = "answer with caveat"
else:
    decision = "ask a clarifying question"
print(decision)

# Chained comparison
print(0.5 <= confidence < 0.8)

# Conditional expression (ternary)
label = "high" if confidence > 0.7 else "low"
print(label)

# Truthiness: empty/zero values are False
docs = []
if not docs:
    print("No documents retrieved, so fall back to web search")

# `and` / `or` SHORT-CIRCUIT and return one of the operands (not always a bool)
user_name = ""
print(user_name or "anonymous")  # "anonymous", a common way to set a default
print(docs and docs[0])  # [] (stops before docs[0], so there's no IndexError)

print()
print("=" * 60)
print("2. MATCH (Python 3.10+)")
print("=" * 60)

def handle(event: dict) -> str:
    match event:
        case {"type": "text", "text": t}:
            return f"Show text: {t}"
        case {"type": "tool_use", "name": "search", "input": {"query": q}}:
            return f"Run web search for {q!r}"
        case {"type": "tool_use", "name": name}:
            return f"Unknown tool: {name}"
        case _:
            return "Ignore"

print(handle({"type": "text", "text": "Hi"}))
print(handle({"type": "tool_use", "name": "search", "input": {"query": "python"}}))
print(handle({"type": "tool_use", "name": "delete_db"}))
print(handle({"type": "ping"}))

print()
print("=" * 60)
print("3. FOR LOOPS")
print("=" * 60)

for i in range(3):  # 0, 1, 2
    print("epoch", i)
print(list(range(2, 10, 3)))  # start, stop (excluded), step -> [2, 5, 8]

for idx, word in enumerate(["plan", "act", "observe"], start=1):
    print(f"step {idx}: {word}")

for q, a in zip(["2+2?", "capital of PK?"], ["4", "Islamabad"]):
    print(q, "->", a)

# UNDER THE HOOD: `for x in obj` really does this:
it = iter(["a", "b"])  # get an iterator
print(next(it), next(it))  # ask for items one by one
try:
    next(it)
except StopIteration:
    print("StopIteration: the for loop catches this and stops")

# Don't modify a list while looping over it. Loop over a copy or build a new list.
nums = [1, 2, 3, 4]
nums = [n for n in nums if n % 2 == 0]
print(nums)

print()
print("=" * 60)
print("4. WHILE, BREAK, CONTINUE, ELSE")
print("=" * 60)

retries = 0
while retries < 5:
    retries += 1
    if retries == 2:
        continue  # skip the rest of this iteration
    print("attempt", retries)
    if retries == 4:
        print("success, so break")
        break

# for-else: `else` runs only if the loop did NOT break
for doc in ["intro", "setup", "faq"]:
    if doc == "pricing":
        print("found pricing")
        break
else:
    print("pricing doc not found")

print()
print("=" * 60)
print("5. COMPREHENSIONS")
print("=" * 60)

# [expression for item in iterable if condition]
squares = [x * x for x in range(6)]
evens = [x for x in range(10) if x % 2 == 0]
cleaned = [w.strip().lower() for w in ["  Hello", "WORLD  ", " "] if w.strip()]
print(squares, evens, cleaned)

# Equivalent long form of `cleaned`:
cleaned_long = []
for w in ["  Hello", "WORLD  ", " "]:
    if w.strip():
        cleaned_long.append(w.strip().lower())
print(cleaned == cleaned_long)

# dict & set comprehensions
scores = {"doc1": 0.9, "doc2": 0.4, "doc3": 0.75}
relevant = {k: v for k, v in scores.items() if v >= 0.7}
first_letters = {w[0] for w in ["agent", "api", "batch"]}
print(relevant, first_letters)

# Nested: flatten a list of chunks
pages = [["p1c1", "p1c2"], ["p2c1"], ["p3c1", "p3c2"]]
flat = [chunk for page in pages for chunk in page]
print(flat)

# Matrix (a list of lists) and transpose, which is a preview of NumPy
matrix = [[1, 2, 3], [4, 5, 6]]
transposed = [[row[i] for row in matrix] for i in range(3)]
print(transposed)

# Generator expression: lazy, uses almost no memory
total = sum(x * x for x in range(1_000_000))  # no list ever gets built
print(total)

# Rule: if a comprehension doesn't fit on about 2 lines, use a normal loop.

print()
print("=" * 60)
print("6. PUTTING IT TOGETHER: a fake agent loop")
print("=" * 60)

# Real agents: call the LLM -> if it asks for a tool, run the tool ->
# feed the result back -> repeat until it gives a final answer (or hits max steps).
fake_llm_replies = [
    {"type": "tool_use", "name": "calculator", "input": {"expr": "17 * 23"}},
    {"type": "tool_use", "name": "search", "input": {"query": "population of Lahore"}},
    {"type": "text", "text": "17*23 = 391, and Lahore has ~14 million people."},
]

MAX_STEPS = 5
history = []
for step in range(MAX_STEPS):
    reply = fake_llm_replies[step]  # pretend this came from an API
    if reply["type"] == "text":
        print(f"[step {step}] FINAL: {reply['text']}")
        break
    tool = reply["name"]
    if tool == "calculator":
        result = eval(reply["input"]["expr"])  # NEVER eval untrusted input in real code!
    elif tool == "search":
        result = "~14 million (fake search result)"
    else:
        result = f"error: unknown tool {tool}"
    print(f"[step {step}] called {tool} -> {result}")
    history.append({"tool": tool, "result": result})
else:
    print("Stopped: hit MAX_STEPS without a final answer")

# ---------------------------------------------------------------
# EXERCISES (exercises/05.py)
# ---------------------------------------------------------------
# 1. FizzBuzz 1..30 (Fizz for multiples of 3, Buzz for 5, FizzBuzz for both).
# 2. Given texts = ["Good movie", "", "bad plot ", None, "GREAT"], use ONE
#    comprehension to get non-empty, stripped, lowercased strings.
# 3. Retry simulation: loop up to 3 times, "calling" a function that fails
#    the first 2 times, and print which attempt succeeded. Use for-else to
#    print "gave up" if all attempts fail.
# 4. Build {word: count} for a sentence using a loop, then again with a
#    dict comprehension (hint: .count on a list).
# 5. Extend the agent loop: add a "weather" tool and a reply that uses it.
