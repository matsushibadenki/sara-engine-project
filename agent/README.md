# SARA Root Agent

This directory contains a lightweight, Codex-inspired local agent for SARA Engine.

It follows the same broad loop as a terminal coding agent: understand the task, retrieve local context, run safe tools, observe results, and respond. It does not call a remote LLM. The current implementation uses project learning data from `data/raw/chat_data.jsonl`, `data/interim/chat_data.jsonl`, and `data/corpus.txt`.

## Run

```bash
python -m agent.cli "Pythonの特徴を教えて"
```

Show the internal loop:

```bash
python -m agent.cli "2 + 3 * 4 を計算して" --show-trace
```

Interactive mode:

```bash
python -m agent.cli --interactive
```

Save a trace:

```bash
python -m agent.cli "Pythonとは？" --save-trace
```

Traces are saved under `workspace/agent/`, which keeps generated outputs inside the repository's managed output policy.
