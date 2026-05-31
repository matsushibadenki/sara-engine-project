"""Command line interface for the root-level SARA agent."""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from .sara_codex_agent import (
    ApprovalMode,
    SaraCodexAgent,
    SaraCodexAgentConfig,
    format_result,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a Codex-inspired local SARA agent over project training data."
    )
    parser.add_argument("task", nargs="*", help="Task or question for the agent.")
    parser.add_argument(
        "--data",
        action="append",
        default=None,
        help="Training data path. Can be passed multiple times.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of training hits to use.")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ApprovalMode],
        default=ApprovalMode.AUTO.value,
        help="Tool execution mode.",
    )
    parser.add_argument("--show-trace", action="store_true", help="Print agent loop steps.")
    parser.add_argument(
        "--save-trace",
        nargs="?",
        const="workspace/agent/last_trace.json",
        default=None,
        help="Save a JSON trace under a managed output path.",
    )
    parser.add_argument("--interactive", action="store_true", help="Start an interactive session.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = SaraCodexAgentConfig(
        data_paths=tuple(args.data) if args.data else SaraCodexAgentConfig().data_paths,
        top_k=max(1, args.top_k),
        approval_mode=ApprovalMode(args.mode),
    )
    agent = SaraCodexAgent(config=config)

    if args.interactive:
        print("SARA Agent ready. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                task = input("Task: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("")
                return 0
            if task.lower() in {"exit", "quit"}:
                return 0
            if not task:
                continue
            result = agent.run(task)
            print(format_result(result, show_trace=args.show_trace))
            if args.save_trace:
                saved = agent.save_trace(result, args.save_trace)
                print(f"Trace saved: {saved}")
        return 0

    task = " ".join(args.task).strip()
    if not task:
        parser.error("task is required unless --interactive is used")

    result = agent.run(task)
    print(format_result(result, show_trace=args.show_trace))
    if args.save_trace:
        saved = agent.save_trace(result, args.save_trace)
        print(f"Trace saved: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
