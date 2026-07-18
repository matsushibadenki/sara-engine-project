#!/usr/bin/env python3
"""Collect multilingual, source-backed Phase 19/20 language evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "phase19_20_language" / "source_documents.jsonl"
PROCESSED_PATH = ROOT / "data" / "processed" / "phase19_20_language" / "heldout_cases.jsonl"
COLLECTION_TIME = "2026-07-18T00:00:00Z"


SOURCES = [
    {
        "record_id": "language-en-pathlib-001",
        "language": "en",
        "source_url": "https://docs.python.org/3/library/pathlib.html",
        "source_revision": "Python 3.14.6 documentation, 2026-07-14",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "content": "This module offers classes representing filesystem paths with semantics appropriate for different operating systems. Path classes are divided between pure paths, which provide purely computational operations without I/O, and concrete paths, which inherit from pure paths but also provide I/O operations.",
    },
    {
        "record_id": "language-en-argparse-001",
        "language": "en",
        "source_url": "https://docs.python.org/3/library/argparse.html",
        "source_revision": "Python 3.14.6 documentation, 2026-07-14",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "content": "The argparse module makes it easy to write user-friendly command-line interfaces. The program defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. The argparse module also automatically generates help and usage messages and issues errors when users give invalid arguments.",
    },
    {
        "record_id": "language-ja-pathlib-001",
        "language": "ja",
        "source_url": "https://docs.python.org/ja/3/library/pathlib.html",
        "source_revision": "Python 3.14.6 Japanese documentation, 2026-07-16",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "content": "このモジュールはファイルシステムのパスを表すクラスを提供していて、様々なオペレーティングシステムについての適切な意味論をそれらのクラスに持たせています。純粋パスは I/O を伴わない純粋な計算操作を提供します。具象パスは純粋パスを継承していますが、I/O 操作も提供しています。",
    },
    {
        "record_id": "language-ja-argparse-001",
        "language": "ja",
        "source_url": "https://docs.python.org/ja/3/library/argparse.html",
        "source_revision": "Python 3.14.6 Japanese documentation, 2026-07-16",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "content": "argparse モジュールはユーザーフレンドリーなコマンドラインインターフェースを簡単に作成します。プログラムは必要な引数を定義し、argparse は sys.argv からそれらを解析します。argparse はヘルプと使用法のメッセージを自動的に生成し、無効な引数が渡されたときにエラーを出します。",
    },
    {
        "record_id": "language-zh-cn-pathlib-001",
        "language": "zh-CN",
        "source_url": "https://docs.python.org/zh-cn/3/library/pathlib.html",
        "source_revision": "Python 3.14.6 Simplified Chinese documentation, 2026-07-11",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "content": "该模块提供表示文件系统路径的类，其语义适用于不同的操作系统。路径类被分为提供纯计算操作而没有 I/O 的纯路径，以及从纯路径继承而来但提供 I/O 操作的具体路径。",
    },
    {
        "record_id": "language-zh-cn-argparse-001",
        "language": "zh-CN",
        "source_url": "https://docs.python.org/zh-cn/3/library/argparse.html",
        "source_revision": "Python 3.14.6 Simplified Chinese documentation, 2026-07-11",
        "license_hint": "Python Documentation License 2.0; https://docs.python.org/3/license.html",
        "content": "argparse 模块让编写用户友好的命令行接口变得容易。程序定义它需要哪些参数，argparse 将会知道如何从 sys.argv 解析它们。argparse 模块还能自动生成帮助和用法消息文本，并在用户传入无效参数时发出错误提示。",
    },
]


TASKS_BY_LANGUAGE = {
    "en": [
        ("qa", "long_distance_dependency", "What does the documented module provide and what is the distinction between its two roles?", "retrieve", ["module"]),
        ("negative", "unsupported_query", "What does this documentation say about GPU tensor backpropagation and transformer weights?", "abstain", ["GPU", "backpropagation"]),
        ("contrastive", "negation_scope", "Does the source describe pure computation without I/O, or does it require I/O for every operation?", "retrieve", ["I/O"]),
        ("delayed", "delayed_recall", "After an intervening unrelated update, recall the source's named module and its main operational distinction.", "retrieve", ["module"]),
    ],
    "ja": [
        ("qa", "long_distance_dependency", "このドキュメントのモジュールは何を提供し、二つの役割にはどのような違いがありますか。", "retrieve", ["モジュール"]),
        ("negative", "unsupported_query", "この文書は GPU テンソルのバックプロパゲーションと Transformer の重みについて何と説明していますか。", "abstain", ["GPU", "バックプロパゲーション"]),
        ("contrastive", "negation_scope", "純粋な計算は I/O を伴わないと説明されていますか、それともすべての操作に I/O が必要ですか。", "retrieve", ["I/O"]),
        ("delayed", "delayed_recall", "無関係な更新を挟んだ後で、文書にあるモジュール名と主な操作上の違いを思い出してください。", "retrieve", ["モジュール"]),
    ],
    "zh-CN": [
        ("qa", "long_distance_dependency", "文档中的模块提供什么功能，两个角色之间有什么区别？", "retrieve", ["模块"]),
        ("negative", "unsupported_query", "这份文档如何说明 GPU 张量反向传播和 Transformer 权重？", "abstain", ["GPU", "反向传播"]),
        ("contrastive", "negation_scope", "来源说明纯计算不需要 I/O，还是每个操作都需要 I/O？", "retrieve", ["I/O"]),
        ("delayed", "delayed_recall", "在插入一次无关更新之后，回忆文档中的模块名称和主要操作区别。", "retrieve", ["模块"]),
    ],
}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build() -> tuple[list[dict], list[dict]]:
    raw: list[dict] = []
    cases: list[dict] = []
    for source in SOURCES:
        source_hash = _hash(source["content"])
        raw.append(
            {
                "schema": "sara-independent-language-source-row-v1",
                **source,
                "source_hash": source_hash,
                "collection_time": COLLECTION_TIME,
                "evidence_scope": "independent_external",
                "observed_only": True,
                "compliance_level": "allow",
                "content_origin": "transcribed_source_excerpt",
            }
        )
        for index, task in enumerate(TASKS_BY_LANGUAGE[source["language"]]):
            task_type, task_family, query, expected_behavior, expected_keywords = task
            case_id = f"{source['record_id']}-{index:02d}"
            cases.append(
                {
                    "schema": "sara-independent-language-case-v1",
                    "case_id": case_id,
                    "language": source["language"],
                    "task_type": task_type,
                    "task_family": task_family,
                    "query": query,
                    "document": source["content"],
                    "expected_keywords": expected_keywords,
                    "expected_behavior": expected_behavior,
                    "source_record_id": source["record_id"],
                    "source_url": source["source_url"],
                    "source_domain": "docs.python.org",
                    "source_hash": source_hash,
                    "source_revision": source["source_revision"],
                    "license_hint": source["license_hint"],
                    "collection_time": COLLECTION_TIME,
                    "evidence_scope": "independent_external",
                    "observed_only": True,
                    "compliance_level": "allow",
                    "derivation_stage": "post_source_split",
                }
            )
    return raw, cases


def main() -> int:
    raw, cases = build()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("w", encoding="utf-8") as handle:
        for row in raw:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with PROCESSED_PATH.open("w", encoding="utf-8") as handle:
        for row in cases:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Collected {len(raw)} multilingual source documents")
    print(f"Derived {len(cases)} post-split held-out cases")
    print(f"Raw output: {RAW_PATH}")
    print(f"Processed output: {PROCESSED_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
