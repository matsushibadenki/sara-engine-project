# ファイルパス: src/sara_engine/utils/data/pinterest.py
# ファイル名: pinterest.py
# ファイルの内容: Pinterestから画像データを収集し、SNN画像分類器のトレーニング用に加工・提供するためのユーティリティ関数とクラスを定義するモジュール。
import json
import os
import re
import time
from dataclasses import dataclass
from html import unescape
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from sara_engine.models.spiking_image_classifier import (
    SNNImageClassifierConfig,
    SpikingImageClassifier,
)
from sara_engine.utils.project_paths import (
    ensure_output_directory,
    ensure_parent_directory,
    raw_data_path,
)


ImageLoader = Callable[[str, Tuple[int, int]], List[float]]


def slugify_label(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    return slug.strip("-") or "pinterest"


def _normalize_candidate_url(candidate: str, base_url: str) -> Optional[str]:
    cleaned = unescape(candidate).replace("\\/", "/").strip()
    if cleaned.startswith("//"):
        cleaned = "https:" + cleaned
    if not cleaned:
        return None
    resolved = urljoin(base_url, cleaned)
    parsed = urlparse(resolved)
    if parsed.scheme not in {"http", "https"}:
        return None
    return resolved


def extract_pinterest_image_urls(html: str, base_url: str) -> List[str]:
    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'"orig"\s*:\s*\{\s*"url"\s*:\s*"([^"]+)"',
        r'"images"\s*:\s*\{.*?"url"\s*:\s*"([^"]+)"',
        r'"image_url"\s*:\s*"([^"]+)"',
        r'<img[^>]+src=["\']([^"\']+)["\']',
        r'<img[^>]+srcset=["\']([^"\']+)["\']',
    ]

    urls: List[str] = []
    seen = set()
    for pattern in patterns:
        for match in re.findall(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            candidates: Iterable[str]
            if "srcset" in pattern:
                candidates = [part.strip().split(" ")[0]
                              for part in match.split(",")]
            else:
                candidates = [match]
            for candidate in candidates:
                resolved = _normalize_candidate_url(candidate, base_url)
                if not resolved:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                urls.append(resolved)
    return urls


def _default_image_loader(image_path: str, image_size: Tuple[int, int]) -> List[float]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required for default Pinterest image preprocessing. "
            "Pass a custom image_loader or install Pillow."
        ) from exc

    with Image.open(image_path) as image:
        resized = image.convert("L").resize(image_size)
        pixels = list(resized.getdata())
    return [pixel / 255.0 for pixel in pixels]


def _guess_extension(url: str, content_type: Optional[str]) -> str:
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1].lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        return suffix
    if content_type:
        normalized = content_type.split(";", 1)[0].strip().lower()
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        return mapping.get(normalized, ".img")
    return ".img"


@dataclass
class PinterestRecord:
    label: str
    source_page_url: str
    image_url: str
    local_path: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "source_page_url": self.source_page_url,
                "image_url": self.image_url,
                "local_path": self.local_path,
            },
            ensure_ascii=False,
        )


class PinterestImageCollector:
    """
    Collects images from public Pinterest pages into managed project paths.

    This helper is intended for pages you are allowed to access and reuse.
    """

    def __init__(self, user_agent: str = "sara-engine-pinterest-loader/0.1") -> None:
        self.user_agent = user_agent

    def _open_bytes(
        self,
        url: str,
        timeout: int,
        opener: Optional[Callable[..., Any]] = None,
    ) -> Tuple[bytes, Optional[str]]:
        request = Request(url, headers={"User-Agent": self.user_agent})
        active_opener = opener or urlopen
        response = active_opener(request, timeout=timeout)
        with response:
            content = response.read()
            headers = getattr(response, "headers", None)
            if headers is None:
                content_type = None
            else:
                content_type = headers.get("Content-Type")
        return content, content_type

    def fetch_public_page(
        self,
        page_url: str,
        timeout: int = 20,
        opener: Optional[Callable[..., Any]] = None,
    ) -> str:
        content, _ = self._open_bytes(page_url, timeout=timeout, opener=opener)
        return content.decode("utf-8", errors="ignore")

    def download_images(
        self,
        page_url: str,
        label: str,
        limit: int = 50,
        output_dir: Optional[str] = None,
        timeout: int = 20,
        opener: Optional[Callable[..., Any]] = None,
        sleep_seconds: float = 0.0,
    ) -> str:
        html = self.fetch_public_page(page_url, timeout=timeout, opener=opener)
        image_urls = extract_pinterest_image_urls(
            html, base_url=page_url)[:limit]
        if not image_urls:
            raise ValueError(f"No image URLs found on page: {page_url}")

        label_slug = slugify_label(label)
        target_dir = output_dir or raw_data_path("pinterest", label_slug)
        images_dir = ensure_output_directory(
            os.path.join(target_dir, "images"))
        manifest_path = ensure_parent_directory(
            os.path.join(target_dir, "manifest.jsonl"))

        records: List[PinterestRecord] = []
        for index, image_url in enumerate(image_urls):
            payload, content_type = self._open_bytes(
                image_url, timeout=timeout, opener=opener)
            extension = _guess_extension(image_url, content_type)
            local_path = os.path.join(images_dir, f"{index:05d}{extension}")
            with open(local_path, "wb") as image_file:
                image_file.write(payload)
            records.append(
                PinterestRecord(
                    label=label,
                    source_page_url=page_url,
                    image_url=image_url,
                    local_path=local_path,
                )
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            for record in records:
                manifest_file.write(record.to_json() + "\n")
        return manifest_path


def load_pinterest_manifest(manifest_path: str) -> List[PinterestRecord]:
    records: List[PinterestRecord] = []
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(PinterestRecord(**payload))
    return records


def build_pinterest_training_samples(
    manifest_paths: Sequence[str],
    label_to_id: Dict[str, int],
    image_size: Tuple[int, int] = (8, 8),
    image_loader: Optional[ImageLoader] = None,
) -> List[Tuple[List[float], int]]:
    active_loader = image_loader or _default_image_loader
    samples: List[Tuple[List[float], int]] = []

    for manifest_path in manifest_paths:
        for record in load_pinterest_manifest(manifest_path):
            if record.label not in label_to_id:
                raise KeyError(f"Unknown label in manifest: {record.label}")
            features = active_loader(record.local_path, image_size)
            expected_size = image_size[0] * image_size[1]
            if len(features) != expected_size:
                raise ValueError(
                    f"Image loader returned {len(features)} features, expected {expected_size}."
                )
            samples.append((features, label_to_id[record.label]))
    return samples


def train_spiking_image_classifier_from_pinterest(
    manifest_paths: Sequence[str],
    label_to_id: Dict[str, int],
    image_size: Tuple[int, int] = (8, 8),
    image_loader: Optional[ImageLoader] = None,
    epochs: int = 3,
) -> SpikingImageClassifier:
    samples = build_pinterest_training_samples(
        manifest_paths=manifest_paths,
        label_to_id=label_to_id,
        image_size=image_size,
        image_loader=image_loader,
    )
    config = SNNImageClassifierConfig(
        input_size=image_size[0] * image_size[1],
        num_classes=len(label_to_id),
    )
    model = SpikingImageClassifier(config)

    for _ in range(epochs):
        for pixels, target_class in samples:
            model.forward(pixels, learning=True, target_class=target_class)
    return model
