# ファイルパス: tests/test_twitter_data.py
# ファイル名: test_twitter_data.py
# ファイルの内容: Twitterデータ収集モジュールのテストコード。画像のURL抽出、データ収集のユニットテストを含む。

from sara_engine.utils.project_paths import workspace_path
from sara_engine.utils.data.twitter import (
    TwitterDataCollector,
    extract_twitter_metadata,
    load_twitter_manifest,
)
import os
import sys
from urllib.request import Request

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../src")))


class _FakeResponse:
    def __init__(self, payload: bytes, content_type: str = "text/html; charset=utf-8"):
        self._payload = payload
        self.headers = {"Content-Type": content_type}

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_extract_twitter_metadata():
    html = """
    <html>
      <head>
        <meta property="og:description" content="SNNの学習にTwitterデータを使ってみるテスト。" />
        <meta property="og:image" content="https://pbs.twimg.com/media/example.jpg" />
        <meta property="og:title" content="AI Researcher (@ai_dev)" />
      </head>
      <body>
      </body>
    </html>
    """

    metadata = extract_twitter_metadata(html)

    assert metadata["text"] == "SNNの学習にTwitterデータを使ってみるテスト。"
    assert metadata["image_url"] == "https://pbs.twimg.com/media/example.jpg"
    assert metadata["author"] == "AI Researcher"


def test_collector_downloads_twitter_page_and_writes_manifest():
    html = """
    <html>
      <head>
        <meta property="og:description" content="AIの進歩は速いですね。" />
        <meta property="og:image" content="https://pbs.twimg.com/media/ai_pic.png" />
        <meta property="og:title" content="Tech News (@tech_news)" />
      </head>
      <body>
      </body>
    </html>
    """.encode("utf-8")
    image_data = b"fake-png-data"

    def fake_opener(request: Request, timeout: int = 20):
        url = request.full_url
        if url == "https://twitter.com/tech_news/status/12345/":
            return _FakeResponse(html)
        if url == "https://pbs.twimg.com/media/ai_pic.png":
            return _FakeResponse(image_data, content_type="image/png")
        raise AssertionError(f"Unexpected URL: {url}")

    collector = TwitterDataCollector()
    output_dir = workspace_path("pytest_twitter_collection")
    manifest_path = collector.download_tweets(
        urls=["https://twitter.com/tech_news/status/12345/"],
        label="AI",
        output_dir=output_dir,
        opener=fake_opener,
        sleep_seconds=0.0,
    )

    records = load_twitter_manifest(manifest_path)
    assert len(records) == 1
    assert records[0].label == "AI"
    assert records[0].author == "Tech News"
    assert records[0].text == "AIの進歩は速いですね。"
    assert os.path.exists(records[0].local_image_path)

    with open(records[0].local_image_path, "rb") as image_file:
        assert image_file.read() == image_data
