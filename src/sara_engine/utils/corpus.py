# ディレクトリパス: src/sara_engine/utils/corpus.py
# ファイルの日本語タイトル: コーパス処理ユーティリティ
# ファイルの目的や内容: テキストデータの前処理、ノイズ除去、行の結合、見出しの分離、箇条書きの正規化、および学習用会話データの生成補助を行う。

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict


_NOISE_SYMBOLS = set("{}[]<>|\\/@#$%^&*_+=~`")
_SENTENCE_ENDERS = "。！？.!?"
_WRAP_CONTINUATION_PREFIXES = (
    "（", "(", "「", "『", "・", ",", "，", ".", "．", "、",
    "を", "が", "に", "で", "と", "は", "も", "の", "や",
)


def is_noisy_line(line: str) -> bool:
    text = line.strip()
    if len(text) < 2:
        return True

    lowered = text.lower()
    if "http://" in lowered or "https://" in lowered or ".pdf" in lowered:
        return True

    noise_count = sum(1 for ch in text if ch in _NOISE_SYMBOLS)
    if noise_count / max(1, len(text)) > 0.18:
        return True

    if re.search(r"[A-Za-z]{6,}\d{2,}", text):
        return True

    jp_chars = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text))
    if len(text) > 0 and (jp_chars / len(text)) < 0.3:
        return True

    return False


def normalize_list_item(line: str) -> str:
    """箇条書きのマーカーを正規化する"""
    line = line.strip()
    line = re.sub(r"^[\-*\+＋・]\s*", "・", line)
    line = re.sub(r"^[0-9０-９]+[\.\)．）]\s*", "・", line)
    return line


def merge_wrapped_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    buffer = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer:
                merged.append(buffer)
                buffer = ""
            continue

        # 箇条書きの正規化
        if _looks_like_list_item(line):
            line = normalize_list_item(line)

        if not buffer:
            buffer = line
            continue

        if _should_merge_lines(buffer, line):
            buffer += line
        else:
            merged.append(buffer)
            buffer = line

    if buffer:
        merged.append(buffer)
    return merged


def clean_corpus_lines(lines: List[str], merge_wrapped: bool = True) -> List[str]:
    cleaned = [line.strip() for line in lines if not is_noisy_line(line)]
    if not merge_wrapped:
        return cleaned
    return merge_wrapped_lines(cleaned)


def _should_merge_lines(prev_line: str, next_line: str) -> bool:
    if not prev_line or not next_line:
        return False

    if prev_line[-1] in _SENTENCE_ENDERS:
        return False

    # 見出しと本文の分離
    if _looks_like_heading(prev_line) or _looks_like_heading(next_line):
        return False
        
    # 箇条書きは結合しない
    if _looks_like_list_item(prev_line) or _looks_like_list_item(next_line):
        return False

    if next_line.startswith(_WRAP_CONTINUATION_PREFIXES):
        return True

    if prev_line.endswith(("、", "（", "(", "・", "第")):
        return True

    if next_line[:1] in "ぁぃぅぇぉゃゅょァィゥェォャュョ":
        return True

    return False


def _looks_like_heading(line: str) -> bool:
    if len(line) <= 3:
        return True
    if line.startswith(("-", "*", "###", "##", "#")):
        return True
    # 数字のみの行も見出しとみなす
    if re.fullmatch(r"[0-9０-９]+", line):
        return True
    return False

def _looks_like_list_item(line: str) -> bool:
    if re.match(r"^[\-*\+＋・]\s+", line):
        return True
    if re.match(r"^[0-9０-９]+[\.\)．）]\s+", line):
        return True
    return False


def generate_conversational_pairs(lines: List[str]) -> List[Tuple[str, str]]:
    """
    説明文コーパスから対話形式（質問->回答など）のデータペアを生成する。
    """
    pairs = []
    for line in lines:
        if not line or len(line) < 10:
            continue
        # 定義の抽出（「〜とは〜である」）
        match = re.search(r"(.+?)(とは|って)(.+?)(です|である|だ)。", line)
        if match:
            term = match.group(1).strip()
            definition = match.group(3).strip() + match.group(4) + "。"
            pairs.append((f"{term}について教えてください。", f"{term}は、{definition}"))
            
        # 途中からの続きを予測させるペア
        midpoint = len(line) // 2
        split_idx = line.find("、", midpoint)
        if split_idx != -1:
            first_half = line[:split_idx + 1]
            second_half = line[split_idx + 1:]
            pairs.append((f"{first_half}の続きを教えてください。", second_half))
    return pairs


def is_low_quality_response(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 8:
        return True
    if len(stripped) > 220:
        return True
    if re.search(r"(Wikipedia|Category:|クリック|右図|pp\.|https?://|\\displaystyle)", stripped):
        return True
    if stripped.startswith(("。", "、", "）", ")", "】", "』", "」")):
        return True
    if re.match(r"^(の|が|を|に|へ|と|で|は|も|や|し|て|な|か|され|する|した|しています|である)", stripped):
        return True
    if stripped.endswith(("（", "(", "、", "・", "の", "が", "を", "に", "は", "と")):
        return True
    if sum(1 for ch in stripped if ch in "{}<>|") >= 1:
        return True
    jp_chars = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", stripped))
    ascii_chars = len(re.findall(r"[A-Za-z]", stripped))
    if jp_chars < 4:
        return True
    if ascii_chars > max(12, jp_chars):
        return True
    return False


def clean_chat_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    cleaned: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for prompt, response in pairs:
        p = re.sub(r"\s+", " ", prompt).strip()
        r = re.sub(r"\s+", " ", response).strip()
        if len(p) < 2 or len(r) < 8:
            continue
        if p == "次の文章に続く言葉を出力してください。":
            continue
        if re.search(r"(続きを教えてください。)$", p) and is_low_quality_response(r):
            continue
        if is_low_quality_response(r):
            continue
        item = (p, r)
        if item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    return cleaned


def load_chat_jsonl_pairs(path: str) -> List[Tuple[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    pairs: List[Tuple[str, str]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data: Dict[str, str] = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = data.get("prompt", "").strip()
            response = data.get("response", data.get("completion", "")).strip()
            if prompt and response:
                pairs.append((prompt, response))
    return clean_chat_pairs(pairs)


def build_definition_qa_pairs(lines: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in lines:
        text = line.strip()
        if len(text) < 20 or len(text) > 180:
            continue

        # "Xは〜である" / "Xとは〜である" 型をQ/A化
        match = re.match(r"(.{2,24}?)(とは|は)、?(.{8,140}?)(です|である|だ|を指す|を意味する)。?$", text)
        if match:
            term = match.group(1).strip("「」『』 ")
            predicate = match.group(3).strip()
            ending = match.group(4)
            if (
                len(term) >= 2
                and not is_noisy_line(term)
                and not re.search(r"[0-9０-９]{2,}|[()（）,:：]|について|これ|それ|ため|よう", term)
                and re.search(r"[\u30A0-\u30FF\u4E00-\u9FFFA-Za-z]", term)
                and not is_low_quality_response(predicate + ending + "。")
            ):
                answer = f"{term}は、{predicate}{ending}。"
                pairs.append((f"{term}とは何ですか？", answer))
                pairs.append((f"{term}について教えてください。", answer))

    return clean_chat_pairs(pairs)
