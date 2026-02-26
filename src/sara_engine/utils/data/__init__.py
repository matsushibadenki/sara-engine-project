_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/data/__init__.py",
    "//": "ファイルの日本語タイトル: データモジュール初期化",
    "//": "ファイルの目的や内容: スパイクベースのデータセットとデータローダーを正しくエクスポートするようインポートエラーを修正。"
}

from .dataset import SpikeDataset, TextSpikeDataset
from .dataloader import SpikeStreamLoader

# 古いコードとの互換性を保つためのフォールバック処理
try:
    from .dataloader import SpikeDataLoader
except ImportError:
    # 存在しない場合は新しいストリームローダーをエイリアスとして割り当てる
    SpikeDataLoader = SpikeStreamLoader

__all__ = [
    "SpikeDataset",
    "TextSpikeDataset",
    "SpikeDataLoader",
    "SpikeStreamLoader"
]