_FILE_INFO = {
    "//": "ディレクトリパス: src/sara_engine/pipelines/token_classification.py",
    "//": "ファイルの日本語タイトル: トークン分類パイプライン",
    "//": "ファイルの目的や内容: バイト単位のNER分類。文頭のエンティティを正しく発火させるため、BOS(文頭)パディングを追加し、デコードを堅牢化する。"
}

from typing import Union, List, Dict
from .base import Pipeline

class TokenClassificationPipeline(Pipeline):
    """
    Token classification (NER) pipeline using pure SNN.
    Groups byte-level predictions into readable entity strings.
    """
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(model, tokenizer, **kwargs)
        self.id2label = kwargs.get("id2label", {0: "O", 1: "PER", 2: "LOC", 3: "ORG"})

    def __call__(self, text_inputs: Union[str, List[str]], **kwargs) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
        if isinstance(text_inputs, str):
            text_inputs = [text_inputs]

        results = []
        for text in text_inputs:
            # 最初の文字が閾値を超えるためのコンテキストとしてスペースをパディング
            padded_text = " " + text
            token_ids = self.tokenizer.encode(padded_text)
            predictions = self.model.forward(token_ids, learning=False)
            
            # パディング部分の予測結果をスキップ
            token_ids = token_ids[1:]
            predictions = predictions[1:]
            
            entities = []
            current_entity = {"label": "O", "bytes": []}
            
            for byte_val, pred_id in zip(token_ids, predictions):
                label = self.id2label.get(pred_id, "O")
                
                if label != current_entity["label"]:
                    if current_entity["label"] != "O" and len(current_entity["bytes"]) > 0:
                        decoded_word = bytes(current_entity["bytes"]).decode('utf-8', errors='ignore').strip()
                        if decoded_word and len(decoded_word) > 0:
                            entities.append({"entity": current_entity["label"], "word": decoded_word})
                    current_entity = {"label": label, "bytes": [byte_val]}
                else:
                    current_entity["bytes"].append(byte_val)
                    
            if current_entity["label"] != "O" and len(current_entity["bytes"]) > 0:
                decoded_word = bytes(current_entity["bytes"]).decode('utf-8', errors='ignore').strip()
                if decoded_word and len(decoded_word) > 0:
                    entities.append({"entity": current_entity["label"], "word": decoded_word})
                
            results.append(entities)

        if len(results) == 1:
            return results[0]
        return results

    def learn(self, text: str, word_labels: List[tuple]) -> None:
        """
        文字列と(単語, ラベルID)のペアを受け取り、バイト単位のアノテーションに変換してSTDP学習を行う。
        """
        padded_text = " " + text
        token_ids = self.tokenizer.encode(padded_text)
        target_classes = [0] * len(token_ids)
        
        for word, label_id in word_labels:
            word_bytes = self.tokenizer.encode(word)
            w_len = len(word_bytes)
            for i in range(len(token_ids) - w_len + 1):
                if token_ids[i:i+w_len] == word_bytes:
                    for j in range(w_len):
                        target_classes[i+j] = label_id
                        
        self.model.forward(token_ids, learning=True, target_classes=target_classes)