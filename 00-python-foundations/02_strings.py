"""
LESSON 02: Strings, the language of LLMs
==========================================
Run:  python 00-python-foundations/02_strings.py

What you'll learn:
  1. Creating strings (quotes, multi-line, escape chars, raw strings)
  2. Indexing and slicing (and how it works inside)
  3. Essential string methods
  4. f-strings (formatting), the tool you'll use for prompt templates
  5. Unicode, encoding, and why "length" isn't "tokens"
"""

print("=" * 60)
print("1. CREATING STRINGS")
print("=" * 60)

single = 'Hello'
double = "It's easy"  # use double quotes when the text has an apostrophe
multi = """You are a helpful assistant.
Answer concisely."""  # triple quotes allow multiple lines, common for system prompts
print(multi)

print("Line1\nLine2\tTabbed")  # \n newline, \t tab
print("She said \"hi\"")  # escaped quote
print(r"C:\new\folder")  # raw string: backslashes are NOT escapes (Windows paths, regex)

print()
print("=" * 60)
print("2. INDEXING & SLICING")
print("=" * 60)

# A string is an immutable SEQUENCE of characters.
#
#   s =   "  A   G   E   N   T  "
#   index   0   1   2   3   4
#   neg    -5  -4  -3  -2  -1
s = "AGENT"
print(s[0], s[-1])  # A T
print(s[1:4])  # GEN   slice [start:stop) where stop is EXCLUDED
print(s[:2], s[2:])  # AG ENT
print(s[::-1])  # TNEGA   step -1 reverses
print(s[::2])  # AET

# Slicing never raises IndexError; indexing does
print(s[2:100])  # "ENT"
# print(s[100])     # IndexError

# Slices create NEW strings (strings are immutable)

print()
print("=" * 60)
print("3. ESSENTIAL METHODS (all return NEW strings)")
print("=" * 60)

raw = "   The Answer Is: 42   \n"
print(repr(raw.strip()))  # remove whitespace at both ends
print(raw.lower(), raw.upper(), sep="|")
print("answer" in raw.lower())  # membership check -> True
print(raw.strip().startswith("The"))  # True
print(raw.strip().replace("42", "forty-two"))
print(raw.find("Answer"), raw.find("zzz"))  # index, or -1 if not found

# split & join, which you'll use constantly
csv_line = "name,age,city"
parts = csv_line.split(",")  # -> list ['name', 'age', 'city']
print(parts)
print(" | ".join(parts))  # list -> string
print("one two   three".split())  # no argument = split on any whitespace

# Parsing a (fake) LLM response
llm_output = "Sentiment: POSITIVE\nConfidence: 0.92"
lines = llm_output.split("\n")
sentiment = lines[0].split(":")[1].strip()
confidence = float(lines[1].split(":")[1])
print(sentiment, confidence, type(confidence))

# Check methods
print("123".isdigit(), "abc".isalpha(), "   ".isspace())

# Performance note: building strings in a loop with += makes a new string
# each time. For many pieces, collect them in a list and "".join() once.
chunks = []
for i in range(5):
    chunks.append(f"chunk-{i}")
print(", ".join(chunks))

print()
print("=" * 60)
print("4. F-STRINGS (formatted string literals)")
print("=" * 60)

user = "Talha"
score = 0.87654
tokens = 1234567

print(f"Hello {user}!")
print(f"Score: {score:.2f}")  # 2 decimal places -> 0.88
print(f"Percent: {score:.1%}")  # 87.7%
print(f"Tokens: {tokens:,}")  # 1,234,567
print(f"[{user:>10}]  [{user:<10}]  [{user:^10}]")  # right/left/center align
print(f"{user=}")  # debug form -> user='Talha'
print(f"Expression: {2 * 21}, upper: {user.upper()}")
print(f"Literal braces: {{not a variable}}")

# Prompt template, how AI engineers actually use f-strings
context = "Python was created by Guido van Rossum in 1991."
question = "Who created Python?"
prompt = f"""Use ONLY the context to answer.

Context:
{context}

Question: {question}
Answer:"""
print(prompt)

# Reusable template with .format() (useful when the template is defined before the data)
TEMPLATE = "Summarize the following text in {n} bullet points:\n{text}"
print(TEMPLATE.format(n=3, text="..."))

print()
print("=" * 60)
print("5. UNICODE, BYTES, AND LENGTH vs TOKENS")
print("=" * 60)

# Python 3 str = Unicode text. Bytes = raw data (files, network).
word = "café 🤖"
print(len(word))  # 6 characters
encoded = word.encode("utf-8")  # str -> bytes
print(encoded, len(encoded))  # more bytes than characters (é = 2 bytes, 🤖 = 4)
print(encoded.decode("utf-8"))  # bytes -> str

# LLMs don't count characters or words, they count TOKENS (sub-word pieces).
# A rough rule for English: 1 token ≈ 4 characters ≈ 0.75 words.
text = "Agentic AI systems plan, use tools, and reflect on results."
print(f"chars={len(text)}, words={len(text.split())}, ~tokens={len(text) // 4}")

# ---------------------------------------------------------------
# AI CONNECTION
# ---------------------------------------------------------------
# - Prompt engineering is building strings with f-strings and templates.
# - Parsing model output uses split/strip/find (later: json and regex).
# - Always use encoding="utf-8" when reading text files for RAG on Windows.
# - Chunking documents for RAG is slicing: text[start:start+chunk_size].

# ---------------------------------------------------------------
# EXERCISES (write answers in exercises/02.py)
# ---------------------------------------------------------------
# 1. Given email = "  Talha.Aslam@Example.COM ", produce "talha.aslam@example.com"
#    and extract the domain "example.com".
# 2. Write a chunker: split a long string into pieces of 20 characters
#    with an overlap of 5 (chunk 1 = [0:20], chunk 2 = [15:35], ...).
#    (Use a while loop, or wait for lesson 05.)
# 3. Build a prompt with f-string that includes a role, a task, and a
#    word limit variable, then print it.
# 4. Parse "name=GPT; params=175B; open=false" into three separate variables.
