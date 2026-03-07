import re
from typing import List


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

    if _looks_like_heading(prev_line) or _looks_like_heading(next_line):
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
    if re.fullmatch(r"[0-9０-９]+", line):
        return True
    return False
