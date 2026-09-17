# 00 — Python Foundations (for Agentic AI, Data Science & AI Engineering)

Learn Python from zero, with each concept tied to the AI work you're aiming for.

## How to study each lesson

1. **Read** the file from top to bottom. The comments are the lesson.
2. **Run** it: `python 00-python-foundations/01_variables_and_types.py`
3. **Predict before you run.** Before each `print`, guess the output. When you guess wrong, that's where you learn.
4. **Change things.** Break the code on purpose and read the error message.
5. **Do the exercises** at the bottom of each file. Write your answers in `exercises/` (e.g. `exercises/01.py`).

Don't copy and paste the examples. Type them out, because typing builds the habit.

## Roadmap

### Phase 1: Core language (this folder, lessons 01–06)
| # | Lesson | Why it matters for AI |
|---|--------|------------------------|
| 01 | Variables, types, memory model | Every tensor, prompt, and API response is an object |
| 02 | Strings & f-strings | Prompts ARE strings. Parsing LLM output is string work |
| 03 | Lists & tuples | Batches, chat histories, token lists |
| 04 | Dictionaries & sets | JSON, API payloads, tool-call arguments, configs |
| 05 | Control flow (if / loops / comprehensions) | Agent loops, data cleaning |
| 06 | Functions | Tools an agent calls are just functions |

### Phase 2: Intermediate Python
07 Error handling · 08 Files & JSON · 09 Modules & packages / venv / pip ·
10 OOP (classes) · 11 Iterators & generators (streaming LLM tokens) ·
12 Decorators · 13 Type hints & dataclasses / Pydantic

### Phase 3: Python for AI engineering
14 HTTP & APIs (`requests`/`httpx`) · 15 `async`/`await` (parallel LLM calls) ·
16 Environment variables & secrets · 17 Logging & testing (`pytest`)

### Phase 4: Data science stack
NumPy → Pandas → Matplotlib → scikit-learn basics

### Phase 5: Agentic AI
LLM API calls → tool/function calling → an agent loop from scratch → RAG
(the `01-`, `02-`, `03-` folders in this repo)

Ask Claude for the next lesson when you finish the exercises for the current one.
