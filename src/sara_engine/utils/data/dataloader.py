_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/data/dataloader.py",
    "//": "ファイルの日本語タイトル: スパイク・ストリーム・データローダー",
    "//": "ファイルの目的や内容: PyTorchのDataLoaderの代替。データセットから連続的な時系列データをリアルタイムでイベント駆動型スパイクとしてエンコードし、非同期ストリームとしてSARAエンジンに供給する。"
}

from typing import Iterator, Tuple, List
from .dataset import SpikeDataset

class SpikeDataLoader:
    """
    SNN特化型のストリームデータローダー。
    静的なテンソルバッチではなく、時間軸に沿ったスパイクイベントのストリームを生成し、
    メモリやCPU負荷を抑えながら学習エンジンへデータを供給する。
    """
    def __init__(self, dataset: SpikeDataset, batch_size: int = 1, shuffle: bool = False):
        # SNNは時間的な連続性と因果関係を重視するため、時系列タスクでは通常シャッフルしない
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Tuple[int, List[int]]]:
        """
        データセットから時間ステップごとにスパイクのリストを取得し、逐次供給する。
        AER (Address-Event Representation) に似た形式でルーティングを行う。
        """
        # 現状はシンプルなジェネレータパススルー。
        # 将来的には複数ストリームの非同期マージ（SNN的なバッチ化）に対応する。
        for timestamp, spikes in self.dataset:
            yield (timestamp, spikes)