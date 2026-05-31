#!/usr/bin/env python3
"""Interactive chat client pinned to autobot production model only."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sara_engine.models.spiking_llm import SpikingLLM  # noqa: E402
from sara_engine.utils.project_paths import model_path  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with autobot production model.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    production_dir = model_path("autobot_self_organized", "production")
    weights_file = os.path.join(production_dir, "spiking_llm_weights.json")

    if not os.path.exists(weights_file):
        print(f"[ERROR] Production model not found: {weights_file}")
        print("Start the bot and wait for a successful promotion first.")
        return 1

    print(f"[INFO] Loading production model: {production_dir}")
    llm = SpikingLLM.from_pretrained(production_dir)
    print("[INFO] Ready. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSARA: Goodbye.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            print("SARA: Goodbye.")
            break

        prompt = f"User: {user_text}\nSARA:"
        response = llm.generate(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            stop_conditions=["\n"],
        )
        print(f"SARA: {str(response).strip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
