_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/utils/data/dataset.py",
    "//": "ファイルの日本語タイトル: データセット基底クラス",
    "//": "ファイルの目的や内容: PyTorchのDatasetに相当する基底クラス。"
}

class Dataset:
    """An abstract class representing a dataset."""
    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

class IterableDataset(Dataset):
    """An iterable dataset for continuous spike streams."""
    def __iter__(self):
        raise NotImplementedError
    
    def __len__(self):
        raise TypeError("IterableDataset does not have a fixed length.")