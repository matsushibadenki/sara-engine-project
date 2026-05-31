"""Utilities for best-effort multimodal ingestion into text training records."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class IngestRecord:
    source_path: str
    modality: str
    content_type: str
    summary_text: str
    metadata: dict[str, object]

    def to_json(self) -> str:
        return json.dumps(
            {
                "source_path": self.source_path,
                "modality": self.modality,
                "content_type": self.content_type,
                "summary_text": self.summary_text,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
        )


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_read_text(path: str) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read(16384).decode("utf-8", errors="ignore")


def _extract_docx_text(path: str) -> str:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", xml)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def _extract_pdf_text(path: str) -> str:
    try:
        with open(path, "rb") as f:
            blob = f.read(2_000_000)
        candidates = re.findall(rb"\(([^\)]{2,400})\)", blob)
        text = " ".join(part.decode("latin-1", errors="ignore") for part in candidates)
        return re.sub(r"\s+", " ", text).strip()
    except Exception:
        return ""


def _audio_metadata(path: str) -> dict[str, object]:
    meta: dict[str, object] = {}
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate) if rate else 0.0
                meta.update(
                    {
                        "channels": wf.getnchannels(),
                        "sample_width": wf.getsampwidth(),
                        "sample_rate": rate,
                        "duration_sec": round(duration, 3),
                    }
                )
        except Exception:
            pass
    return meta


def ingest_file(path: str) -> IngestRecord:
    file_path = os.path.abspath(path)
    ext = Path(file_path).suffix.lower()
    content_type, _ = mimetypes.guess_type(file_path)
    content_type = content_type or "application/octet-stream"
    file_size = os.path.getsize(file_path)
    base_meta: dict[str, object] = {
        "ext": ext,
        "size_bytes": file_size,
        "sha256": _sha256(file_path),
    }

    if content_type.startswith("text/") or ext in {".md", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".xml", ".html", ".py", ".js", ".ts", ".swift", ".java", ".c", ".cpp", ".rs", ".go", ".sh"}:
        text = _safe_read_text(file_path)
        summary = re.sub(r"\s+", " ", text)[:4000]
        return IngestRecord(file_path, "text", content_type, summary, base_meta)

    if ext in {".docx"}:
        text = _extract_docx_text(file_path)
        summary = text[:4000] if text else "[docx] text extraction failed"
        return IngestRecord(file_path, "document", content_type, summary, base_meta)

    if ext in {".pdf"}:
        text = _extract_pdf_text(file_path)
        summary = text[:4000] if text else "[pdf] best-effort text extraction failed"
        return IngestRecord(file_path, "document", content_type, summary, base_meta)

    if content_type.startswith("image/"):
        summary = f"Image asset: {Path(file_path).name} ({content_type})."
        return IngestRecord(file_path, "image", content_type, summary, base_meta)

    if content_type.startswith("audio/") or ext in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
        base_meta.update(_audio_metadata(file_path))
        summary = f"Audio asset: {Path(file_path).name} ({content_type})."
        return IngestRecord(file_path, "audio", content_type, summary, base_meta)

    if content_type.startswith("video/"):
        summary = f"Video asset: {Path(file_path).name} ({content_type})."
        return IngestRecord(file_path, "video", content_type, summary, base_meta)

    summary = f"Binary asset: {Path(file_path).name} ({content_type})."
    return IngestRecord(file_path, "binary", content_type, summary, base_meta)


def ingest_many(paths: Iterable[str]) -> list[IngestRecord]:
    records: list[IngestRecord] = []
    for p in paths:
        try:
            if os.path.isfile(p):
                records.append(ingest_file(p))
        except Exception:
            continue
    return records
