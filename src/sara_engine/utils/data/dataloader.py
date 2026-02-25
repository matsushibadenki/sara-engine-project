_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/data/dataloader.py",
    "//": "ファイルの日本語タイトル: スパイク・データローダー",
    "//": "ファイルの目的や内容: SNN向けにデータをストリーム(スパイクのリスト)として提供するデータローダー。"
}

import random
from .dataset import Dataset, IterableDataset

class SpikeDataLoader:
    """
    Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    Unlike PyTorch, it does not stack data into dense matrices (tensors). 
    Instead, it yields lists of spikes (events) sequentially or in pseudo-batches (lists of lists) 
    to preserve spatial and temporal sparsity.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        if isinstance(self.dataset, IterableDataset):
            # ストリーム型データセットの場合
            batch = []
            for item in self.dataset:
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield batch if self.batch_size > 1 else batch[0]
                    batch = []
            if batch:
                yield batch if self.batch_size > 1 else batch[0]
        else:
            # 静的データセットの場合
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
                
            batch = []
            for idx in indices:
                batch.append(self.dataset[idx])
                if len(batch) == self.batch_size:
                    yield batch if self.batch_size > 1 else batch[0]
                    batch = []
            if batch:
                yield batch if self.batch_size > 1 else batch[0]

    def __len__(self):
        if isinstance(self.dataset, IterableDataset):
            raise TypeError("Cannot determine length of DataLoader with IterableDataset.")
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size