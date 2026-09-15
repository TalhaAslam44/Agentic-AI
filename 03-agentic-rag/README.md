# Project 3 — Agentic RAG, built step by step

Take the retriever from Project 2, connect it to Claude, and grow it from a
fixed pipeline into an agent that decides for itself when and what to search.

Built in small steps. Each step has a file I write that teaches the concept,
and an exercise you write that proves you have it. Do not skip an exercise.

## Setup

Get an API key at console.anthropic.com, then set it for your terminal session:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Never put the key in a file inside this repo.

## The steps

| Step | You learn | Agentic concept |
|---|---|---|
| 1 | Messages, roles, stop reasons, tokens, cost | The model is stateless |
| 2 | Naive RAG: retrieve, then stuff the prompt | Workflow: code decides every step |
| 3 | Grounding, citations, saying "I don't know" | Guardrails |
| 4 | Evaluating answers, LLM as judge | You cannot improve what you do not measure |
| 5 | Retrieval as a tool the model can call | Tools and action |
| 6 | The agent loop, written by hand | Perceive, reason, act, repeat |
| 7 | Multi-step search, iteration limits, budgets | Planning and autonomy limits |
| 8 | Tracing, then naive RAG versus agentic RAG | Observability, and when agents are worth it |

## Step 1 files

- `step1_first_call.py` — read it, run it, study the output
- `exercise1_chat.py` — fill in four TODOs to make a chatbot with memory
