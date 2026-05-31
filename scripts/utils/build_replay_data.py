import json
import os
from typing import Any, Dict, Iterable, List, Optional

from sara_engine.utils.project_paths import ensure_parent_directory, workspace_path


def _load_tokenizer_runtime():
    from transformers import AutoTokenizer

    return AutoTokenizer


def _iter_chat_examples(data_path: str) -> Iterable[Dict[str, str]]:
    with open(data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue

            if isinstance(payload.get("tokens"), list) and all(isinstance(item, int) for item in payload["tokens"]):
                yield {"tokens": payload["tokens"], "source": "pretokenized"}
                continue
            if isinstance(payload.get("input_ids"), list) and all(isinstance(item, int) for item in payload["input_ids"]):
                yield {"tokens": payload["input_ids"], "source": "pretokenized"}
                continue

            prompt = payload.get("prompt") or payload.get("user")
            response = payload.get("response") or payload.get("completion") or payload.get("sara")
            if isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip():
                text = f"You: {prompt}\nSARA: {response}\n"
                yield {"text": text, "source": "chat_pair"}


def build_replay_data(
    data_path: str,
    output_path: str,
    tokenizer_name: str = "google/gemma-2-2b",
) -> Dict[str, Any]:
    resolved_data_path = os.path.abspath(data_path)
    if not os.path.exists(resolved_data_path):
        raise FileNotFoundError(f"Replay source data not found: {resolved_data_path}")

    AutoTokenizer = _load_tokenizer_runtime()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    resolved_output_path = ensure_parent_directory(output_path)
    example_count = 0
    token_count = 0

    with open(resolved_output_path, "w", encoding="utf-8") as handle:
        for example in _iter_chat_examples(resolved_data_path):
            tokens = example.get("tokens")
            if isinstance(tokens, list):
                token_ids = [int(item) for item in tokens]
            else:
                text = str(example.get("text", ""))
                batch = tokenizer(text, return_tensors="pt")
                raw_input_ids = batch["input_ids"][0]
                if hasattr(raw_input_ids, "tolist"):
                    token_ids = [int(item) for item in raw_input_ids.tolist()]
                else:
                    token_ids = [int(item) for item in raw_input_ids]

            if not token_ids:
                continue

            token_count += len(token_ids)
            example_count += 1
            handle.write(
                json.dumps(
                    {
                        "tokens": token_ids,
                        "source": str(example.get("source", "chat_pair")),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "input_path": resolved_data_path,
        "output_path": resolved_output_path,
        "tokenizer_name": tokenizer_name,
        "example_count": example_count,
        "token_count": token_count,
    }


def default_replay_output_path() -> str:
    return workspace_path("replay", "chat_replay_tokens.jsonl")
