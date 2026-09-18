"""
LESSON 01: Variables, Data Types, and Python's Memory Model
==========================================================
Run:  python 00-python-foundations/01_variables_and_types.py

What you'll learn:
  1. What a variable REALLY is (a name pointing at an object)
  2. Core types: int, float, bool, str, None
  3. type(), id(), isinstance()
  4. Mutable vs immutable (the most important idea in this lesson)
  5. Type conversion (casting)
  6. Arithmetic operators and a float gotcha
"""

print("=" * 60)
print("1. VARIABLES ARE NAMES, NOT BOXES")
print("=" * 60)

# In many languages a variable is a "box" that holds a value.
# In Python, a variable is a NAME (a label) attached to an OBJECT in memory.
#
#     age ──────► [ int object: 25 ]
#
# Every object has three things:
#   - identity : where it lives in memory   -> id(obj)
#   - type     : what kind of thing it is   -> type(obj)
#   - value    : the data itself

age = 25
print("value:", age)
print("type :", type(age))
print("id   :", id(age))  # a memory address-like number

# Two names can point at the SAME object:
a = [1, 2, 3, 4]
b = a  # b does NOT copy the list; it's a second label on the same object
b.append(4)
print("a after b.append(4):", a)  # [1, 2, 3, 4]  <- surprised? a and b are the same list
print("a is b:", a is b)  # `is` checks identity (same object)
print("a == b:", a == b)  # `==` checks value (equal contents)

print()
print("=" * 60)
print("2. CORE DATA TYPES")
print("=" * 60)

count = 100  # int: whole numbers, unlimited size in Python
temperature = 0.7  # float: decimals (the LLM "temperature" setting is a float!)
is_ready = True  # bool: True / False (capitalized)
model_name = "claude"  # str: text
result = None  # NoneType: "no value yet"

for value in [count, temperature, is_ready, model_name, result]:
    print(f"{value!r:>10} -> {type(value).__name__}")

# Python ints never overflow:
print("2 ** 100 =", 2**100)

# bool is actually a subclass of int (True == 1, False == 0)
print("True + True =", True + True)  # 2, handy for counting: sum([True, False, True]) == 2

# None is a single object. Always compare with `is`, not `==`
response = None
if response is None:
    print("No response from the model yet")

# isinstance() is the right way to check a type (it respects inheritance)
print("isinstance(True, int):", isinstance(True, int))  # True
print("isinstance(0.7, (int, float)):", isinstance(0.7, (int, float)))

print()
print("=" * 60)
print("3. MUTABLE vs IMMUTABLE  (very important)")
print("=" * 60)

# IMMUTABLE: int, float, bool, str, tuple, None. The object can't change.
# MUTABLE:   list, dict, set, most custom objects. The object CAN change in place.

x = 10
print("id(x) before:", id(x))
x = x + 1  # this does NOT modify 10; it creates a NEW object 11 and moves the label
print("id(x) after :", id(x), "(different object)")

name = "agent"
# name[0] = "A"   # uncomment: TypeError, str is immutable
name = "A" + name[1:]  # build a NEW string instead
print(name)

nums = [1, 2, 3]
print("id(nums) before:", id(nums))
nums.append(4)  # modifies the SAME object
print("id(nums) after :", id(nums), "(same object)")


# Why you care (a real bug in AI code):
def add_message(msg, history=[]):  # BAD: default list is created ONCE and reused
    history.append(msg)
    return history


print(add_message("hi"))  # ['hi']
print(add_message("hello"))  # ['hi', 'hello']  <- memory leaked between conversations!


def add_message_fixed(msg, history=None):  # GOOD
    if history is None:
        history = []  # a fresh list on every call
    history.append(msg)
    return history


print(add_message_fixed("hi"))  # ['hi']
print(add_message_fixed("hello"))  # ['hello']

print()
print("=" * 60)
print("4. TYPE CONVERSION (CASTING)")
print("=" * 60)

user_input = "42"  # input() and files always give you strings
print(int(user_input) + 8)  # 50
print(float("3.14"))  # 3.14
print(str(99) + " tokens")  # "99 tokens"
print(int(7.9))  # 7  (truncates, doesn't round)
print(round(7.9))  # 8
print(bool(0), bool(""), bool([]), bool(None))  # all False ("falsy")
print(bool(1), bool("x"), bool([0]))  # all True ("truthy")

try:
    int("forty-two")
except ValueError as e:
    print("ValueError:", e)  # LLMs often return bad numbers, so always validate

print()
print("=" * 60)
print("5. OPERATORS")
print("=" * 60)

print("7 / 2  =", 7 / 2)  # 3.5  true division ALWAYS returns float
print("7 // 2 =", 7 // 2)  # 3    floor division
print("7 % 2  =", 7 % 2)  # 1    remainder (modulo)
print("2 ** 3 =", 2**3)  # 8    power
print("-7 // 2 =", -7 // 2)  # -4   floors toward negative infinity!

# Float gotcha: computers store floats in binary, so some decimals are approximate
print("0.1 + 0.2 =", 0.1 + 0.2)  # 0.30000000000000004
print("0.1 + 0.2 == 0.3:", 0.1 + 0.2 == 0.3)  # False!
import math

print("math.isclose:", math.isclose(0.1 + 0.2, 0.3))  # True, use this for floats

# Augmented assignment
tokens_used = 0
tokens_used += 150
tokens_used *= 2
print("tokens_used:", tokens_used)

# Multiple assignment and swapping
prompt_tokens, completion_tokens = 120, 80
prompt_tokens, completion_tokens = completion_tokens, prompt_tokens  # swap, no temp var
print(prompt_tokens, completion_tokens)

# ---------------------------------------------------------------
# AI CONNECTION
# ---------------------------------------------------------------
# - The mutable default bug above shows up for real in chatbot code,
#   where one user's chat history ends up in another user's session.
# - Float precision matters in ML: model weights are float32/float16,
#   and comparing probabilities with == is almost always a bug.
# - Casting and validating is exactly what you do with LLM output:
#   the model returns "0.87" as text, and you must turn it into float(0.87).

# ---------------------------------------------------------------
# EXERCISES (write answers in 00-python-foundations/exercises/01.py)
# ---------------------------------------------------------------
# 1. Create variables for an LLM call: model (str), max_tokens (int),
#    temperature (float), stream (bool). Print each with its type.
# 2. Predict the output, then run it:
#        x = [1, 2]; y = x; y = y + [3]; print(x)
#    Why is it different from using y.append(3)? (Hint: + creates a new object.)
# 3. Cost calculator: input price is $3 per 1,000,000 tokens and output is
#    $15 per 1,000,000. For 12,500 input and 3,200 output tokens, print
#    the total cost rounded to 6 decimals.
# 4. Given raw = "  0.75 " (with spaces), convert it to a float safely.
#    (Hint: strings have a .strip() method.)
