import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sara_engine.utils.project_paths import workspace_path


def _load_replay_module():
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "scripts", "utils", "build_replay_data.py")
    )
    spec = importlib.util.spec_from_file_location("build_replay_data_script", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_replay_data_tokenizes_chat_pairs_into_managed_jsonl(monkeypatch):
    module = _load_replay_module()
    data_path = workspace_path("tests", "replay_source_chat.jsonl")
    output_path = workspace_path("tests", "replay_tokens.jsonl")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"user": "hi", "sara": "hello"}) + "\n")
        handle.write(json.dumps({"prompt": "bye", "response": "goodbye"}) + "\n")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

        def __call__(self, text: str, return_tensors: str = "pt"):
            return {"input_ids": [[ord(char) for char in text]]}

    monkeypatch.setattr(module, "_load_tokenizer_runtime", lambda: FakeTokenizer)

    report = module.build_replay_data(data_path, output_path, tokenizer_name="stub")

    assert report["example_count"] == 2
    assert report["token_count"] > 0
    assert report["output_path"] == os.path.abspath(output_path)

    with open(output_path, "r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    assert len(lines) == 2
    assert all("tokens" in item for item in lines)
    assert all(item["source"] == "chat_pair" for item in lines)


def test_build_replay_data_preserves_pretokenized_examples(monkeypatch):
    module = _load_replay_module()
    data_path = workspace_path("tests", "replay_source_tokens.jsonl")
    output_path = workspace_path("tests", "replay_tokens_passthrough.jsonl")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    with open(data_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"tokens": [1, 2, 3]}) + "\n")

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, _name):
            return cls()

    monkeypatch.setattr(module, "_load_tokenizer_runtime", lambda: FakeTokenizer)

    report = module.build_replay_data(data_path, output_path, tokenizer_name="stub")

    assert report["example_count"] == 1
    with open(output_path, "r", encoding="utf-8") as handle:
        payload = json.loads(handle.readline())
    assert payload["tokens"] == [1, 2, 3]
    assert payload["source"] == "pretokenized"
