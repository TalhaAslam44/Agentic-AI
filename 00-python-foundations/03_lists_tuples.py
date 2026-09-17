"""
LESSON 03: Lists and Tuples
=============================
Run:  python 00-python-foundations/03_lists_tuples.py

What you'll learn:
  1. Lists: creating, indexing, modifying
  2. How a list works inside (dynamic array) and Big-O costs
  3. Sorting, copying (shallow vs deep)
  4. Tuples, unpacking, and when to use a tuple over a list
"""
import copy

print("=" * 60)
print("1. LISTS: ordered, mutable, can hold anything")
print("=" * 60)

models = ["claude", "llama", "mistral"]
mixed = [1, "two", 3.0, True, None, [5, 6]]  # allowed, but usually keep one type
print(models, len(models))
print(models[0], models[-1], models[0:2])

models[1] = "gemma"  # modify in place
models.append("phi")  # add to end
models.insert(0, "gpt")  # insert at index
models.extend(["qwen", "deepseek"])  # add many
print(models)

last = models.pop()  # remove & return last
first = models.pop(0)  # remove & return at index
models.remove("phi")  # remove by value (first match)
print(last, first, models)
print("claude" in models, models.index("mistral"), models.count("claude"))

print()
print("=" * 60)
print("2. UNDER THE HOOD: a list is a dynamic array of POINTERS")
print("=" * 60)

# A list stores references (pointers) to objects, packed side by side.
# It keeps extra spare slots so append() is usually instant.
#
#   list ─► [ ptr | ptr | ptr | spare | spare ]
#             │     │     │
#            obj   obj   obj
#
# Operation cost (n = length):
#   lst[i], lst[i] = x, append, pop()   -> O(1)  fast
#   insert(0, x), pop(0), remove(x)     -> O(n)  shifts every element
#   x in lst                            -> O(n)  scans one by one
# So: don't use a list as a queue (use collections.deque), and
# don't do heavy membership tests on lists (use a set, lesson 04).
import sys

growing = []
for i in range(10):
    growing.append(i)
    print(f"len={len(growing):2}  bytes={sys.getsizeof(growing)}")  # jumps = over-allocation

print()
print("=" * 60)
print("3. SORTING & COPYING")
print("=" * 60)

scores = [0.72, 0.95, 0.31, 0.88]
print(sorted(scores))  # returns NEW list
print(sorted(scores, reverse=True))
scores.sort()  # sorts IN PLACE, returns None
print(scores)

# Sort by a key: this pattern is everywhere (ranking RAG results!)
docs = [("doc_a", 0.72), ("doc_b", 0.95), ("doc_c", 0.31)]
ranked = sorted(docs, key=lambda d: d[1], reverse=True)
print("top-2:", ranked[:2])

# Copying
original = [[1, 2], [3, 4]]
alias = original  # same object
shallow = original.copy()  # new outer list, SAME inner lists (also: original[:])
deep = copy.deepcopy(original)  # everything copied

original[0].append(99)
print("alias  :", alias)  # [[1, 2, 99], [3, 4]]
print("shallow:", shallow)  # [[1, 2, 99], [3, 4]]  <- inner list is shared!
print("deep   :", deep)  # [[1, 2], [3, 4]]

# Gotcha: [[]] * 3 makes 3 references to ONE inner list
grid = [[]] * 3
grid[0].append("x")
print(grid)  # [['x'], ['x'], ['x']]
grid = [[] for _ in range(3)]  # correct way
grid[0].append("x")
print(grid)

# Useful built-ins
nums = [4, 8, 15, 16, 23, 42]
print(sum(nums), min(nums), max(nums), sum(nums) / len(nums))
for i, n in enumerate(nums[:3]):  # index + value
    print(i, n)
names = ["q", "k", "v"]
dims = [64, 64, 128]
print(list(zip(names, dims)))  # pair up -> [('q', 64), ...]

print()
print("=" * 60)
print("4. TUPLES: ordered, IMMUTABLE")
print("=" * 60)

point = (3, 4)
single = (5,)  # the comma makes a tuple, not the parentheses: (5) is just int 5
print(type(single), type((5)))

# point[0] = 10   # TypeError

# Unpacking
x, y = point
print(x, y)
first, *rest = [1, 2, 3, 4]  # star-unpacking
print(first, rest)

# Functions return multiple values as a tuple
def min_max(values):
    return min(values), max(values)

lo, hi = min_max([3, 9, 1])
print(lo, hi)

# When to use a tuple:
#  - a fixed "record" whose parts mean different things: (name, score)
#  - as dict keys / set members (lists can't be, since they're unhashable)
#  - shapes in NumPy/PyTorch: array.shape == (batch, seq_len, hidden)
shape = (32, 512, 768)
batch, seq_len, hidden = shape
print(f"batch={batch} seq_len={seq_len} hidden={hidden}")

# Tuples are slightly smaller & faster than lists
print(sys.getsizeof([1, 2, 3]), sys.getsizeof((1, 2, 3)))

# ---------------------------------------------------------------
# AI CONNECTION
# ---------------------------------------------------------------
# Chat history for an LLM API is a list of messages:
chat = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "Retrieval-Augmented Generation..."},
]
chat.append({"role": "user", "content": "Give an example."})
print(f"{len(chat)} messages, last role: {chat[-1]['role']}")
# Keep only the last N messages to fit the context window:
print(chat[-2:])

# ---------------------------------------------------------------
# EXERCISES (exercises/03.py)
# ---------------------------------------------------------------
# 1. Given results = [("a", 0.4), ("b", 0.9), ("c", 0.7), ("d", 0.95)],
#    print the names of the top-3 by score.
# 2. Write batches(items, size) that returns a list of lists, e.g.
#    batches([1,2,3,4,5], 2) -> [[1,2],[3,4],[5]]. (Used for embedding APIs.)
# 3. Explain (in a comment) why a = [1,2]; b = a[:]; b.append(3) leaves a unchanged.
# 4. Trim a chat list so it keeps the first message (system prompt) plus the
#    last 4 messages.
