# ファイルパス: src/sara_engine/utils/data/twitter.py
# ファイル名: twitter.py
# ファイルの内容: X (Twitter) からのポストのテキストや画像データを収集し、学習用にフォーマットするユーティリティクラスおよび関数を定義するモジュール。

import json
import os
import re
import time
from dataclasses import dataclass
from html import unescape
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from sara_engine.utils.project_paths import (
    ensure_output_directory,
    ensure_parent_directory,
    raw_data_path,
)


def slugify_label(value: str) -> str:
    lowered = value.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    return slug.strip("-") or "twitter"


def extract_twitter_metadata(html: str) -> Dict[str, str]:
    """
    HTMLまたはJSONからツイートのテキストや画像を抽出する。
    X(旧Twitter)のOGPタグやSyndication APIのJSONの構造に対応する。
    """
    metadata: Dict[str, str] = {
        "text": "",
        "image_url": "",
        "author": "",
    }

    # 1. 共通のOGPメタタグからの抽出
    description_match = re.search(
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    if description_match:
        metadata["text"] = unescape(description_match.group(1)).strip()

    image_match = re.search(
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    if image_match:
        url = unescape(image_match.group(1)).strip()
        # プロフィール画像等を除外する簡易フィルタ
        if "profile_images" not in url:
            metadata["image_url"] = url

    title_match = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^\(]+)\(@[^"\']+\)["\']', html, flags=re.IGNORECASE)
    if title_match:
        metadata["author"] = unescape(title_match.group(1)).strip()

    # 2. JSONペイロード(Syndication APIなど)からの抽出 (フォールバック)
    if not metadata["text"]:
        text_match = re.search(r'"text"\s*:\s*"((?:\\.|[^"\\])*)"', html)
        if text_match:
            try:
                # unicodeエスケープを処理
                parsed_text = json.loads(f'"{text_match.group(1)}"')
                metadata["text"] = parsed_text.strip()
            except Exception:
                pass

    return metadata


@dataclass
class TwitterRecord:
    label: str
    source_url: str
    text: str
    author: str = ""
    image_url: str = ""
    local_image_path: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "source_url": self.source_url,
                "text": self.text,
                "author": self.author,
                "image_url": self.image_url,
                "local_image_path": self.local_image_path,
            },
            ensure_ascii=False,
        )


class TwitterDataCollector:
    """
    X (Twitter) からポスト情報を取得し、データセット化するコレクター。
    主に公開APIやSyndicationページから情報収集を行うための支援クラス。
    """

    def __init__(self, user_agent: str = "sara-engine-twitter-loader/0.1") -> None:
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

    def _guess_extension(self, url: str) -> str:
        parsed = urlparse(url)
        suffix = os.path.splitext(parsed.path)[1].lower()
        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
            return suffix
        return ".jpg"

    def download_tweets(
        self,
        urls: List[str],
        label: str,
        output_dir: Optional[str] = None,
        timeout: int = 20,
        opener: Optional[Callable[..., Any]] = None,
        sleep_seconds: float = 1.0,
    ) -> str:
        """
        ツイートURLのリストからメタ情報を抽出し、文字・画像データを保存する。
        ツイートのHTMLパースによる抽出を試みる。
        """
        if not urls:
            raise ValueError("提供されたURLリストが空です。")

        label_slug = slugify_label(label)
        target_dir = output_dir or raw_data_path("twitter", label_slug)
        images_dir = ensure_output_directory(
            os.path.join(target_dir, "images"))
        manifest_path = ensure_parent_directory(
            os.path.join(target_dir, "manifest.jsonl"))

        records: List[TwitterRecord] = []
        for index, text_url in enumerate(urls):
            try:
                html = self.fetch_public_page(
                    text_url, timeout=timeout, opener=opener)
                meta = extract_twitter_metadata(html)

                # テキスト情報が取得できなければスキップ
                if not meta.get("text"):
                    continue

                record = TwitterRecord(
                    label=label,
                    source_url=text_url,
                    text=meta["text"],
                    author=meta.get("author", ""),
                )

                image_url = meta.get("image_url")
                if image_url:
                    record.image_url = image_url
                    ext = self._guess_extension(image_url)
                    local_path = os.path.join(images_dir, f"{index:05d}{ext}")

                    try:
                        img_payload, _ = self._open_bytes(
                            image_url, timeout=timeout, opener=opener)
                        with open(local_path, "wb") as img_file:
                            img_file.write(img_payload)
                        record.local_image_path = local_path
                    except (HTTPError, URLError):
                        pass

                records.append(record)

                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            except (HTTPError, URLError):
                print(f"URLからのデータ取得に失敗しました: {text_url}")
                continue

        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            for record in records:
                manifest_file.write(record.to_json() + "\n")

        return manifest_path


def load_twitter_manifest(manifest_path: str) -> List[TwitterRecord]:
    """保存されたTwitterのJSONLマニフェストファイルを読み込む。"""
    records: List[TwitterRecord] = []
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(TwitterRecord(**payload))
    return records
