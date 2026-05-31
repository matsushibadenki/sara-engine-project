# Directory Path: src/sara_engine/cli.py
# English Title: SARA CLI Entry Points
# Purpose/Content: Provides lightweight, testable `sara-chat` and `sara-train`
# entry points with managed path validation for model outputs.

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional, Sequence

from .utils.project_paths import ensure_parent_directory, model_path, resolve_project_relative


def build_chat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SARA Hippocampus Chat Engine")
    parser.add_argument(
        "--model",
        type=str,
        default=model_path("distilled_sara_llm.msgpack"),
        help="Path to the learned memory model.",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=3, help="Top-K candidate limit")
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.2,
        help="Biological refractory period penalty",
    )
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generated tokens")
    parser.add_argument(
        "--save-on-exit",
        action="store_true",
        help="Persist updated diagnostics and memory state back to the managed model path when the session exits.",
    )
    parser.add_argument(
        "--show-diagnostics",
        action="store_true",
        help="Print recent retrieval diagnostics after each response when supported by the runtime.",
    )
    return parser


def build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SARA Dialogue Trainer")
    parser.add_argument(
        "data",
        type=str,
        help="Path to JSONL training data (for example data/raw/chat_data.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=model_path("distilled_sara_llm.msgpack"),
        help="Managed output path for the trained model memory.",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="google/gemma-2-2b",
        help="Teacher model identifier used for distillation guidance.",
    )
    parser.add_argument(
        "--student-sdr-size",
        type=int,
        default=8192,
        help="Sparse distributed representation size for the student model.",
    )
    parser.add_argument(
        "--student-vocab-size",
        type=int,
        default=256000,
        help="Vocabulary capacity for the student model.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum teacher sequence length per dialogue example.",
    )
    return parser


def _parse_chat_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_chat_parser().parse_args(argv)


def _parse_train_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_train_parser().parse_args(argv)


def _validate_existing_file(path: str, label: str) -> str:
    resolved = resolve_project_relative(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _validate_training_output(path: str) -> str:
    return ensure_parent_directory(path)


def _maybe_format_chat_diagnostics(runtime: Any, enabled: bool) -> str:
    if not enabled or not hasattr(runtime, "format_recent_retrieval_diagnostics"):
        return ""
    try:
        diagnostics = runtime.format_recent_retrieval_diagnostics(limit=3)
    except TypeError:
        diagnostics = runtime.format_recent_retrieval_diagnostics()
    return diagnostics if isinstance(diagnostics, str) else ""


def _persist_chat_runtime(runtime: Any, model_file: str) -> bool:
    if not hasattr(runtime, "save_pretrained"):
        return False
    runtime.save_pretrained(model_file)
    return True


def run_chat_cli(argv: Optional[Sequence[str]] = None) -> int:
    from .inference import SaraInference

    args = _parse_chat_args(argv)
    try:
        model_file = _validate_existing_file(args.model, "Model file")
    except FileNotFoundError as exc:
        print(f"[Error] {exc}")
        print("Please check the execution directory.")
        return 1

    print("Loading SARA Engine...")
    sara = SaraInference(model_path=model_file)

    print("Ready! Type 'quit' or 'exit' to stop.")
    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.strip().lower() in ["quit", "exit"]:
            print("SARA: Goodbye! Let's talk again.")
            break

        if not user_input.strip():
            continue

        sara.reset_state()
        start_time = time.time()
        prompt = f"You: {user_input}\nSARA:"

        response = sara.generate(
            prompt,
            max_new_tokens=args.max_length,
            top_k=args.top_k,
            temperature=args.temperature,
            refractory_penalty=args.repetition_penalty,
            stop_conditions=["\n"],
        )

        elapsed_time = time.time() - start_time

        if not response:
            print("SARA: (I have no memory of this)")
        else:
            clean_response = response.replace("\n", "")
            print(f"SARA: {clean_response}  [time {elapsed_time:.3f}s]")
        diagnostics_text = _maybe_format_chat_diagnostics(sara, args.show_diagnostics)
        if diagnostics_text:
            print(diagnostics_text)
    if args.save_on_exit:
        persisted = _persist_chat_runtime(sara, model_file)
        if persisted:
            print(f"[INFO] Saved updated SARA state to {model_file}")
    return 0


def _load_chat_lines(data_path: str) -> list[dict]:
    chat_lines = []
    with open(data_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chat_lines.append(json.loads(line))
    return chat_lines


def _load_training_runtime() -> tuple[Any, Any, Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    import tqdm

    return AutoModelForCausalLM, AutoTokenizer, torch, tqdm


def _build_student_model(args: argparse.Namespace):
    from .models.spiking_llm import SpikingLLM

    return SpikingLLM(
        num_layers=2,
        sdr_size=args.student_sdr_size,
        vocab_size=args.student_vocab_size,
    )


def _resolve_device(torch_module: Any) -> str:
    return "mps" if torch_module.backends.mps.is_available() else (
        "cuda" if torch_module.cuda.is_available() else "cpu"
    )


def _load_or_initialize_student_memory(student: Any, model_output: str) -> None:
    if os.path.exists(model_output):
        print(f"Opening SNN memory file: {model_output}...")
        loaded_count = student.load_memory(model_output)
        print(f"Loaded {loaded_count} patterns.")
    else:
        print("[Warning] No existing memory found. Creating new brain...")
        student._direct_map = {}


def _normalize_token_ids(batch_input_ids: Any) -> list[int]:
    first_row = batch_input_ids[0]
    if hasattr(first_row, "tolist"):
        return list(first_row.tolist())
    return [int(token_id) for token_id in first_row]


def _update_direct_memory_from_teacher(
    student: Any,
    input_ids: list[int],
    probs: Any,
    torch_module: Any,
) -> None:
    context_tokens: list[int] = []
    usable_steps = min(max(0, len(input_ids) - 1), len(probs))
    for idx in range(usable_steps):
        context_tokens.append(input_ids[idx])
        if len(context_tokens) > 8:
            context_tokens.pop(0)

        sdr_key = student._sdr_key(student._encode_to_sdr(context_tokens))
        if sdr_key not in student._direct_map:
            student._direct_map[sdr_key] = {}

        direct_map = student._direct_map[sdr_key]
        actual = input_ids[idx + 1]
        direct_map[actual] = direct_map.get(actual, 0.0) + 500.0

        top_probs, top_indices = torch_module.topk(probs[idx], 5)
        for rank in range(5):
            token_index = int(top_indices[rank].item())
            if token_index != actual:
                direct_map[token_index] = direct_map.get(token_index, 0.0) + (
                    50.0 * top_probs[rank].item()
                )

        for token_id in list(direct_map.keys()):
            if token_id != actual:
                direct_map[token_id] *= 0.8
            if direct_map[token_id] > 1000.0:
                direct_map[token_id] = 1000.0


def run_train_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_train_args(argv)
    try:
        data_path = _validate_existing_file(args.data, "Training data")
        model_output = _validate_training_output(args.model)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[Error] {exc}")
        return 1

    try:
        AutoModelForCausalLM, AutoTokenizer, torch, tqdm = _load_training_runtime()
    except ImportError as exc:
        print(f"[Error] Missing training dependency: {exc}")
        return 1

    print(f"Initializing SNN Student Model ({args.student_sdr_size} neurons)...")
    student = _build_student_model(args)
    device = _resolve_device(torch)

    print(f"Loading teacher model: {args.teacher_model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=torch.float32,
        device_map=device,
    )
    teacher.eval()

    _load_or_initialize_student_memory(student, model_output)

    chat_lines = _load_chat_lines(data_path)
    print(f"Start learning {len(chat_lines)} dialogue items...")

    for item in tqdm.tqdm(chat_lines, desc="Chat Training"):
        text = f"You: {item['user']}\nSARA: {item['sara']}\n"
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_length,
        ).to(device)
        input_ids = _normalize_token_ids(inputs["input_ids"])
        if len(input_ids) < 2:
            continue

        with torch.no_grad():
            outputs = teacher(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        _update_direct_memory_from_teacher(student, input_ids, probs, torch)

    print("Saving updated memory...")
    student.save_memory(model_output)
    print("Dialogue training completed successfully!")
    return 0


def chat(argv: Optional[Sequence[str]] = None) -> int:
    """Start interactive dialogue from terminal with SARA."""
    return run_chat_cli(argv)


def train(argv: Optional[Sequence[str]] = None) -> int:
    """Train SARA dialogue memory from JSONL data."""
    return run_train_cli(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Module entry point.

    `python -m sara_engine.cli --model ...` defaults to chat mode for convenience.
    `python -m sara_engine.cli train ...` runs the training entry point.
    """
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if args and args[0] == "train":
        return run_train_cli(args[1:])
    if args and args[0] == "chat":
        return run_chat_cli(args[1:])
    return run_chat_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
