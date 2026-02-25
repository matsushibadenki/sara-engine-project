_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/data/dataset.py",
    "//": "ファイルの日本語タイトル: スパイクデータセット基底クラス",
    "//": "ファイルの目的や内容: 従来の静的なバッチデータを、SNN向けの連続的なイベントストリーム（スパイク）に変換するための基底クラス群。"
}

from typing import Iterator, Tuple, List

class SpikeDataset:
    """
    SNNエンジン向けのデータセット抽象基底クラス。
    """
    def __init__(self):
        pass

    def __iter__(self) -> Iterator[Tuple[int, List[int]]]:
        """
        時間ステップごとに発火したニューロン（または特徴量）のIDリストを返す。
        戻り値: (timestamp, [active_feature_ids])
        """
        raise NotImplementedError("Subclasses must implement __iter__")

class TextSpikeDataset(SpikeDataset):
    """
    自然言語のトークンIDリストを、時間軸に沿ったスパイクストリームに変換するデータセット。
    """
    def __init__(self, text_ids: List[int], time_step_per_token: int = 1):
        super().__init__()
        self.text_ids = text_ids
        self.time_step_per_token = time_step_per_token

    def __iter__(self) -> Iterator[Tuple[int, List[int]]]:
        current_time = 0
        for token_id in self.text_ids:
            # 1つのトークンIDを特定の時刻におけるスパイクとして発火させる
            yield (current_time, [token_id])
            current_time += self.time_step_per_token