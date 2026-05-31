# ファイルパス: tests/test_pinterest_data.py
# ファイル名: test_pinterest_data.py
# ファイルの内容: Pinterestデータ収集・学習ユーティリティモジュールのテストコード。画像のURL抽出、データ収集、学習サンプルの構築機能などのユニットテストを含む。
from sara_engine.utils.project_paths import workspace_path
from sara_engine.utils.data.pinterest import (
    PinterestImageCollector,
    build_pinterest_training_samples,
    extract_pinterest_image_urls,
    load_pinterest_manifest,
    train_spiking_image_classifier_from_pinterest,
)
import json
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


def test_extract_pinterest_image_urls_handles_meta_json_and_srcset():
    html = """
    <html>
      <head>
        <meta property="og:image" content="https://i.pinimg.com/736x/aa/bb/cc/example.jpg" />
        <script type="application/json">
          {"orig": {"url": "https:\\/\\/i.pinimg.com\\/originals\\/11\\/22\\/33\\/another.png"}}
        </script>
      </head>
      <body>
        <img srcset="//i.pinimg.com/236x/foo.webp 1x, //i.pinimg.com/474x/foo.webp 2x" />
      </body>
    </html>
    """

    urls = extract_pinterest_image_urls(
        html, base_url="https://www.pinterest.com/pin/example/")

    assert "https://i.pinimg.com/736x/aa/bb/cc/example.jpg" in urls
    assert "https://i.pinimg.com/originals/11/22/33/another.png" in urls
    assert "https://i.pinimg.com/236x/foo.webp" in urls
    assert "https://i.pinimg.com/474x/foo.webp" in urls


def test_collector_downloads_public_page_assets_and_writes_manifest():
    html = b"""
    <meta property=\"og:image\" content=\"https://i.pinimg.com/736x/sample-a.jpg\" />
    <img src=\"https://i.pinimg.com/originals/sample-b.png\" />
    """
    image_a = b"fake-jpg"
    image_b = b"fake-png"

    def fake_opener(request: Request, timeout: int = 20):
        url = request.full_url
        if url == "https://www.pinterest.com/pin/abc/":
            return _FakeResponse(html)
        if url == "https://i.pinimg.com/736x/sample-a.jpg":
            return _FakeResponse(image_a, content_type="image/jpeg")
        if url == "https://i.pinimg.com/originals/sample-b.png":
            return _FakeResponse(image_b, content_type="image/png")
        raise AssertionError(f"Unexpected URL: {url}")

    collector = PinterestImageCollector()
    output_dir = workspace_path("pytest_pinterest_collection")
    manifest_path = collector.download_images(
        page_url="https://www.pinterest.com/pin/abc/",
        label="Cats",
        limit=2,
        output_dir=output_dir,
        opener=fake_opener,
    )

    records = load_pinterest_manifest(manifest_path)
    assert len(records) == 2
    assert records[0].label == "Cats"
    assert os.path.exists(records[0].local_path)
    assert os.path.exists(records[1].local_path)

    with open(records[0].local_path, "rb") as image_file:
        assert image_file.read() == image_a
    with open(records[1].local_path, "rb") as image_file:
        assert image_file.read() == image_b


def test_training_helpers_build_samples_and_train_classifier():
    output_dir = workspace_path("pytest_pinterest_training")
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, "sample.bin")
    with open(image_path, "wb") as image_file:
        image_file.write(b"payload")

    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        manifest_file.write(
            json.dumps(
                {
                    "label": "cats",
                    "source_page_url": "https://www.pinterest.com/pin/abc/",
                    "image_url": "https://i.pinimg.com/736x/sample-a.jpg",
                    "local_path": image_path,
                }
            )
            + "\n"
        )

    def fake_image_loader(local_path: str, image_size):
        assert local_path == image_path
        assert image_size == (4, 4)
        return [1.0] * 16

    samples = build_pinterest_training_samples(
        [manifest_path],
        label_to_id={"cats": 0},
        image_size=(4, 4),
        image_loader=fake_image_loader,
    )

    assert samples == [([1.0] * 16, 0)]

    model = train_spiking_image_classifier_from_pinterest(
        [manifest_path],
        label_to_id={"cats": 0},
        image_size=(4, 4),
        image_loader=fake_image_loader,
        epochs=2,
    )
    assert model.config.input_size == 16
    assert model.config.num_classes == 1
