"""
LESSON 06: Functions, the building block of agent tools
=========================================================
Run:  python 00-python-foundations/06_functions.py

What you'll learn:
  1. Defining, calling, returning
  2. Parameters: positional, keyword, defaults, *args, **kwargs, / and *
  3. Scope (LEGB) and how arguments are passed
  4. Functions are objects: pass them around, store them in dicts
  5. lambda, closures
  6. Docstrings + type hints, then build a tool registry for an agent
"""

print("=" * 60)
print("1. BASICS")
print("=" * 60)

def greet(name):
    """Return a greeting."""  # docstring: describes the function
    return f"Hello, {name}!"

print(greet("Talha"))

def no_return():
    x = 1  # no `return` statement...

print(no_return())  # ...so it returns None

def divide(a, b):
    if b == 0:
        return None  # early return
    return a / b

print(divide(10, 4), divide(1, 0))

print()
print("=" * 60)
print("2. PARAMETERS")
print("=" * 60)

def call_llm(prompt, model="claude-sonnet-5", temperature=0.7, max_tokens=512):
    return f"[{model} t={temperature} max={max_tokens}] {prompt}"

print(call_llm("Hi"))
print(call_llm("Hi", temperature=0.0))  # keyword argument: clear & order-free
print(call_llm(prompt="Hi", max_tokens=50, model="claude-haiku-4-5"))

# *args collects extra positional args into a TUPLE
def average(*numbers):
    return sum(numbers) / len(numbers) if numbers else 0.0

print(average(1, 2, 3, 4))

# **kwargs collects extra keyword args into a DICT
def log_event(event, **metadata):
    print(event, metadata)

log_event("tool_call", tool="search", latency_ms=120)

# Unpacking INTO a call
params = {"temperature": 0.1, "max_tokens": 100}
print(call_llm("Summarize", **params))
print(average(*[10, 20, 30]))

# `/` = everything before it is positional-only; `*` = everything after is keyword-only
def embed(text, /, *, normalize=True, dim=768):
    return f"embed({text!r}, normalize={normalize}, dim={dim})"

print(embed("hello", dim=384))
# embed(text="hello")    -> TypeError (positional-only)
# embed("hello", False)  -> TypeError (keyword-only)

print()
print("=" * 60)
print("3. SCOPE & ARGUMENT PASSING")
print("=" * 60)

# Name lookup order is LEGB: Local -> Enclosing -> Global -> Built-in
counter = 0  # global

def bump():
    global counter  # needed to REASSIGN a global (usually avoid this)
    counter += 1

bump(); bump()
print("counter:", counter)

# Python passes REFERENCES to objects ("pass by object reference"):
def mutate(lst):
    lst.append("changed")  # modifies the caller's object

def rebind(lst):
    lst = ["new"]  # only rebinds the LOCAL name, so the caller is unaffected

data = ["orig"]
mutate(data)
rebind(data)
print(data)  # ['orig', 'changed']

print()
print("=" * 60)
print("4. FUNCTIONS ARE OBJECTS")
print("=" * 60)

def shout(s):
    return s.upper() + "!"

def whisper(s):
    return s.lower() + "..."

speak = shout  # no parentheses = the function itself, not a call
print(speak("hey"), speak.__name__)

def apply_all(text, funcs):
    return [f(text) for f in funcs]

print(apply_all("Hello", [shout, whisper, len]))

print()
print("=" * 60)
print("5. LAMBDA & CLOSURES")
print("=" * 60)

# lambda = a small anonymous function with a single expression
square = lambda x: x * x
print(square(7))
people = [("ali", 31), ("sara", 24), ("omar", 28)]
print(sorted(people, key=lambda p: p[1]))

# Closure: an inner function that remembers variables from its enclosing scope
def make_prompt_template(role):
    def template(task):
        return f"You are a {role}. Task: {task}"
    return template

teacher = make_prompt_template("patient Python teacher")
reviewer = make_prompt_template("strict code reviewer")
print(teacher("explain loops"))
print(reviewer("review this PR"))

# Recursion: a function calling itself (needs a base case)
def count_nested(obj):
    """Count leaf values in nested dicts/lists (like JSON)."""
    if isinstance(obj, dict):
        return sum(count_nested(v) for v in obj.values())
    if isinstance(obj, list):
        return sum(count_nested(v) for v in obj)
    return 1  # base case

print(count_nested({"a": 1, "b": [2, 3, {"c": 4}]}))  # 4

print()
print("=" * 60)
print("6. TYPE HINTS, DOCSTRINGS & AN AGENT TOOL REGISTRY")
print("=" * 60)

# Type hints don't change how the code runs. They document it and let editors
# and tools (mypy, Pydantic, LLM SDKs) check it or generate schemas from it.
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get the current weather for a city.

    Args:
        city: City name, e.g. "Lahore".
        unit: "celsius" or "fahrenheit".
    """
    fake_temp = 34 if unit == "celsius" else 93
    return {"city": city, "temp": fake_temp, "unit": unit}

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

print(get_weather.__doc__.splitlines()[0])
print(get_weather.__annotations__)

# The core of every agent framework: map tool NAMES (what the LLM says) to FUNCTIONS
TOOLS = {
    "get_weather": get_weather,
    "add": add,
}

def run_tool(name: str, arguments: dict):
    func = TOOLS.get(name)
    if func is None:
        return {"error": f"unknown tool {name}"}
    try:
        return func(**arguments)  # **kwargs unpacking, from section 2
    except TypeError as e:
        return {"error": f"bad arguments: {e}"}

# Pretend the LLM produced these tool calls:
print(run_tool("get_weather", {"city": "Lahore"}))
print(run_tool("add", {"a": 2, "b": 40}))
print(run_tool("add", {"x": 1}))
print(run_tool("launch_rocket", {}))

# Auto-generate a tool description from the function itself (frameworks do this)
import inspect

def describe(func) -> dict:
    sig = inspect.signature(func)
    return {
        "name": func.__name__,
        "description": inspect.getdoc(func).splitlines()[0],
        "parameters": {
            p.name: getattr(p.annotation, "__name__", str(p.annotation))
            for p in sig.parameters.values()
        },
    }

for f in TOOLS.values():
    print(describe(f))

# ---------------------------------------------------------------
# EXERCISES (exercises/06.py)
# ---------------------------------------------------------------
# 1. Write chunk_text(text: str, size: int = 100, overlap: int = 20) -> list[str]
#    with a docstring. Raise ValueError if overlap >= size.
# 2. Write build_messages(system: str, *user_turns: str) -> list[dict]
#    that returns [{"role":"system",...}, {"role":"user",...}, ...].
# 3. Write make_counter() that returns a function; each call returns 1, 2, 3...
#    (hint: closure + `nonlocal`).
# 4. Add a "word_count" tool to TOOLS and call it through run_tool.
# 5. Write cosine_similarity(a: list[float], b: list[float]) -> float using
#    only built-ins (sum, zip, ** 0.5). This is the core math of semantic search!
