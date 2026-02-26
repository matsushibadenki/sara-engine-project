_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/data/dataloader.py",
    "//": "ファイルの日本語タイトル: スパイク・ストリーム・データローダー",
    "//": "ファイルの目的や内容: PyTorch DataLoaderの代替。静的なバッチではなく、AER (Address-Event Representation) 形式の連続ストリームとして時系列データをエンジンに供給する。"
}

from typing import List, Generator, Any, Callable

class SpikeStreamLoader:
    """
    Biological alternative to PyTorch DataLoader.
    Converts sequential data into an asynchronous Spike Stream (AER format).
    """
    def __init__(self, dataset: List[Any], encode_fn: Callable[[Any], List[int]], time_step: int = 1):
        """
        Args:
            dataset: Raw sequence data (e.g., list of characters, frames, etc.)
            encode_fn: Function to map raw data to spike IDs.
            time_step: Simulated time increment per item.
        """
        self.dataset = dataset
        self.encode_fn = encode_fn
        self.time_step = time_step
        
    def stream(self) -> Generator[dict, None, None]:
        """
        Yields event-driven spikes continuously.
        Returns: { "time": int, "spikes": List[int], "raw": Any }
        """
        current_time = 0
        for item in self.dataset:
            spikes = self.encode_fn(item)
            yield {
                "time": current_time,
                "spikes": spikes,
                "raw": item
            }
            current_time += self.time_step