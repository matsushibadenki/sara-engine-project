"""Semantics-preserving bounded cache for a frozen SARA BPE tokenizer."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def _canonical_fingerprint_payload(
    *,
    vocab: Mapping[str, int],
    merge_ranks: Mapping[Tuple[str, str], int],
    pretokenizer: str,
    special_tokens: Sequence[str],
) -> Dict[str, Any]:
    return {
        "schema": "sara-tokenizer-fingerprint-v1",
        "vocab": [
            [str(token), int(token_id)]
            for token, token_id in sorted(
                vocab.items(), key=lambda item: (int(item[1]), str(item[0]))
            )
        ],
        "merges": [
            [str(pair[0]), str(pair[1]), int(rank)]
            for pair, rank in sorted(
                merge_ranks.items(),
                key=lambda item: (
                    int(item[1]),
                    str(item[0][0]),
                    str(item[0][1]),
                ),
            )
        ],
        "pretokenizer": pretokenizer,
        "normalization": "none",
        "special_tokens": [str(token) for token in special_tokens],
        "unknown_token": "<unk>",
        "implementation": "sara-python-bpe-v1",
    }


def tokenizer_fingerprint(tokenizer: Any) -> str:
    """Fingerprint every field that can change tokenization semantics."""
    pretokenizer = (
        tokenizer.pretokenizer_identity()
        if hasattr(tokenizer, "pretokenizer_identity")
        else f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    )
    payload = _canonical_fingerprint_payload(
        vocab=dict(tokenizer.vocab),
        merge_ranks=dict(tokenizer.merge_ranks),
        pretokenizer=str(pretokenizer),
        special_tokens=tuple(tokenizer.special_tokens),
    )
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BoundedExactTokenizerAdapter:
    """Cache exact pretoken results under hard logical state ceilings."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_entries: int = 1024,
        max_state_bytes: int = 1_048_576,
        max_tokens_per_entry: int = 256,
    ) -> None:
        for name, value in (
            ("max_entries", max_entries),
            ("max_state_bytes", max_state_bytes),
            ("max_tokens_per_entry", max_tokens_per_entry),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.max_entries = max_entries
        self.max_state_bytes = max_state_bytes
        self.max_tokens_per_entry = max_tokens_per_entry
        self._pre_tokenize = tokenizer.pre_tokenize
        self._vocab = dict(tokenizer.vocab)
        self._id_to_token = {
            int(token_id): str(token)
            for token, token_id in self._vocab.items()
        }
        self._merge_ranks = dict(tokenizer.merge_ranks)
        self._special_tokens = tuple(tokenizer.special_tokens)
        self.fingerprint = tokenizer_fingerprint(tokenizer)
        self._unknown_id = int(self._vocab.get("<unk>", 1))
        self._cache: OrderedDict[
            Tuple[str, bytes], Tuple[Tuple[int, ...], int]
        ] = OrderedDict()
        self._state_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._bypassed = 0

    def _tokenize_pretoken(self, pretoken: str) -> Tuple[int, ...]:
        symbols = list(pretoken)
        while len(symbols) > 1:
            pairs = [
                (symbols[index], symbols[index + 1])
                for index in range(len(symbols) - 1)
            ]
            best_pair = min(
                pairs,
                key=lambda pair: self._merge_ranks.get(pair, float("inf")),
            )
            if best_pair not in self._merge_ranks:
                break
            merged: List[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and (symbols[index], symbols[index + 1]) == best_pair
                ):
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return tuple(
            int(self._vocab.get(symbol, self._unknown_id))
            for symbol in symbols
        )

    def _entry_size(
        self,
        key: Tuple[str, bytes],
        token_ids: Tuple[int, ...],
    ) -> int:
        payload = {
            "fingerprint": key[0],
            "pretoken_hex": key[1].hex(),
            "token_ids": list(token_ids),
        }
        return len(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )

    def _lookup(self, pretoken: str) -> Tuple[int, ...]:
        key = (self.fingerprint, pretoken.encode("utf-8"))
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            self._hits += 1
            return cached[0]
        self._misses += 1
        token_ids = self._tokenize_pretoken(pretoken)
        entry_size = self._entry_size(key, token_ids)
        if (
            len(token_ids) > self.max_tokens_per_entry
            or entry_size > self.max_state_bytes
        ):
            self._bypassed += 1
            return token_ids
        while self._cache and (
            len(self._cache) >= self.max_entries
            or self._state_bytes + entry_size > self.max_state_bytes
        ):
            _, (_, removed_size) = self._cache.popitem(last=False)
            self._state_bytes -= removed_size
            self._evictions += 1
        self._cache[key] = (token_ids, entry_size)
        self._state_bytes += entry_size
        return token_ids

    def encode(self, text: str) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        output: List[int] = []
        for pretoken in self._pre_tokenize(text):
            output.extend(self._lookup(pretoken))
        return output

    def encode_utf8(self, payload: bytes) -> List[int]:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        return self.encode(payload.decode("utf-8", errors="strict"))

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(
            self._id_to_token.get(int(token_id), "<unk>")
            for token_id in token_ids
        )

    def clear(self, *, reset_counters: bool = False) -> None:
        self._cache.clear()
        self._state_bytes = 0
        if reset_counters:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._bypassed = 0

    def stats(self) -> Dict[str, Any]:
        return {
            "schema": "sara-bounded-pretoken-cache-stats-v1",
            "fingerprint": self.fingerprint,
            "entries": len(self._cache),
            "state_bytes": self._state_bytes,
            "max_entries": self.max_entries,
            "max_state_bytes": self.max_state_bytes,
            "max_tokens_per_entry": self.max_tokens_per_entry,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "bypassed": self._bypassed,
        }
